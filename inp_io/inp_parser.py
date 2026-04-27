#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inp_parser.py — 解析 Abaqus .inp 到 AssemblyModel
新增：解析 *Surface Interaction / *Contact Property 与 *Friction，
并把 *Contact Pair, interaction=... 正确挂到每对接触；摩擦系数写入
AssemblyModel.interactions[name].friction_mu 供后续查询。

保持：ELSET/NSET 宽松解析、ELEMENT-surface 规范化、常见 3D 单元面提取。
"""

from __future__ import annotations
import os, sys, re, json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Sequence
import numpy as np

# ---- SurfaceDef 导入（兼容 surfaces.py 或 assembly/surfaces.py）----
_CUR = os.path.dirname(__file__)
_SRC = os.path.abspath(os.path.join(_CUR, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SurfSurfaceDef = None
for _mod in ("surfaces", "assembly.surfaces"):
    try:
        SurfSurfaceDef = getattr(__import__(_mod, fromlist=["SurfaceDef"]), "SurfaceDef")
        break
    except Exception:
        pass
if SurfSurfaceDef is None:
    raise ModuleNotFoundError("Cannot import SurfaceDef from 'surfaces' or 'assembly.surfaces'.")

SurfaceDef = SurfSurfaceDef
__all__ = ["SurfaceDef"]

# ======================= 数据结构 =======================
@dataclass
class ElementBlock:
    elem_type: str
    elem_ids: List[int]
    connectivity: List[List[int]]
    raw_params: Dict[str, str] = field(default_factory=dict)

@dataclass
class PartMesh:
    name: str
    node_ids: List[int] = field(default_factory=list)
    nodes_xyz: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    element_blocks: List[ElementBlock] = field(default_factory=list)

@dataclass
class ContactPair:
    master: str
    slave: str
    interaction: Optional[str] = None
    raw: str = ""

@dataclass
class TieConstraint:
    master: str
    slave: str
    raw: str = ""

@dataclass
class BoundaryEntry:
    raw: str

@dataclass
class LoadEntry:
    raw: str

@dataclass
class InstanceDef:
    instance: str
    part: str

@dataclass
class SetDef:
    name: str
    kind: str                 # 'nset' | 'elset'
    scope: str                # 'part' | 'assembly'
    owner: Optional[str]
    items: List[str] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)
    params: Dict[str, str] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

@dataclass
class InteractionProp:
    name: str
    friction_mu: Optional[float] = None
    raw_blocks: List[str] = field(default_factory=list)  # 收集到的子关键字原文（可调试）

@dataclass
class AssemblyModel:
    parts: Dict[str, PartMesh] = field(default_factory=dict)
    surfaces: Dict[str, SurfSurfaceDef] = field(default_factory=dict)
    contact_pairs: List[ContactPair] = field(default_factory=list)
    ties: List[TieConstraint] = field(default_factory=list)
    boundaries: List[BoundaryEntry] = field(default_factory=list)
    loads: List[LoadEntry] = field(default_factory=list)
    nsets: Dict[str, SetDef] = field(default_factory=dict)
    elsets: Dict[str, SetDef] = field(default_factory=dict)
    instances: List[InstanceDef] = field(default_factory=list)

    interactions: Dict[str, InteractionProp] = field(default_factory=dict)  # 新增：接触性质表

    nodes: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    elements: Dict[int, List[int]] = field(default_factory=dict)
    element_types: Dict[int, str] = field(default_factory=dict)

    # --------- Summary ---------
    def summary(self) -> Dict[str, Any]:
        return {
            "num_parts": len(self.parts),
            "parts": list(self.parts.keys()),
            "num_surfaces": len(self.surfaces),
            "surface_names": list(self.surfaces.keys())[:20],
            "num_contact_pairs": len(self.contact_pairs),
            "num_ties": len(self.ties),
            "num_boundaries": len(self.boundaries),
            "num_loads": len(self.loads),
            "num_nsets": len(self.nsets),
            "num_elsets": len(self.elsets),
            "num_instances": len(self.instances),
            "num_interactions": len(self.interactions),   # 新增
            "num_nodes_flat": len(self.nodes),
            "num_elements_flat": len(self.elements),
        }

    # --------- helpers ---------
    @staticmethod
    def _dequote(s: str) -> str:
        return s.strip().strip('"').strip("'")

    @staticmethod
    def _aliases(name: str) -> List[str]:
        n = AssemblyModel._dequote(name)
        cand = [name, n, f'"{n}"', n.lower(), n.upper()]
        nospc = n.replace(" ", "")
        cand += [nospc, f'"{nospc}"', nospc.lower(), nospc.upper()]
        out, seen = [], set()
        for c in cand:
            if c not in seen:
                seen.add(c); out.append(c)
        return out

    @staticmethod
    def _strip_suffix_S(token: str) -> str:
        t = AssemblyModel._dequote(str(token))
        return re.sub(r'([_\.\-:])S(\d+)$', '', t, flags=re.IGNORECASE).strip()

    def expand_elset(self, name: str) -> Sequence[int]:
        raw = name
        base = self._strip_suffix_S(raw)

        sdef: Optional[SetDef] = None
        for k in self._aliases(base):
            if k in self.elsets:
                sdef = self.elsets[k]; break
        if sdef is None:
            deq = self._dequote(raw)
            for k in self._aliases(deq):
                if k in self.elsets:
                    sdef = self.elsets[k]; break
        if sdef is None:
            want = self._dequote(base).replace(" ", "").lower()
            near = []
            for k in self.elsets.keys():
                kk = self._dequote(k).replace(" ", "").lower()
                if want in kk or kk in want:
                    near.append(k)
            hint = f"；候选：{near[:6]}" if near else ""
            raise KeyError(f"ELSET not found: {name}{hint}")

        toks = sdef.items[:]
        params_lower = {k.lower(): v for k, v in (sdef.params or {}).items()}
        is_generate = ('generate' in (sdef.flags or [])) or ('generate' in params_lower)

        def _is_int_token(s: str) -> bool:
            try: int(float(s.strip())); return True
            except Exception: return False

        def _to_int(x: str) -> int:
            return int(float(x.strip()))

        if is_generate or (len(toks) % 3 == 0 and toks and all(_is_int_token(t) for t in toks)):
            out: List[int] = []
            for i in range(0, len(toks), 3):
                try:
                    a = _to_int(toks[i]); b = _to_int(toks[i+1]); c = _to_int(toks[i+2])
                except Exception:
                    continue
                if c == 0: c = 1
                rng = range(a, b + 1, c) if a <= b else range(a, b - 1, -abs(c))
                out.extend(list(rng))
            return out
        else:
            return [_to_int(t) for t in toks if _is_int_token(t)]

    def get_face_nodes(self, elem_id: int, face_id: int) -> np.ndarray:
        """返回所需面的顶点坐标 (k,3)。支持 C3D8/20/4/10/6/15。"""
        eid = int(elem_id)
        if eid not in self.elements:
            raise KeyError(f"Element {eid} not found")
        conn = self.elements[eid]
        n = len(conn)

        def _coords(idxs: List[int]) -> np.ndarray:
            nids = [conn[i] for i in idxs]
            return np.asarray([self.nodes[nid] for nid in nids], dtype=np.float32)

        if n == 8:  # C3D8
            face_idx = {1:[0,1,2,3], 2:[4,5,6,7], 3:[0,4,5,1],
                        4:[1,5,6,2], 5:[2,6,7,3], 6:[3,7,4,0]}
            return _coords(face_idx[int(face_id)])
        if n == 20: # C3D20
            face_idx = {1:[0,1,2,3, 8,9,10,11], 2:[4,5,6,7, 12,13,14,15],
                        3:[0,4,5,1, 16,12,17,8], 4:[1,5,6,2, 17,13,18,9],
                        5:[2,6,7,3, 18,14,19,10], 6:[3,7,4,0, 19,15,16,11]}
            return _coords(face_idx[int(face_id)])
        if n == 4:   # C3D4
            face_idx = {1:[0,1,2], 2:[0,3,1], 3:[1,3,2], 4:[2,3,0]}
            return _coords(face_idx[int(face_id)])
        if n == 10:  # C3D10
            face_idx = {1:[0,1,2, 4,5,6], 2:[0,3,1, 7,8,4],
                        3:[1,3,2, 8,9,5], 4:[2,3,0, 9,7,6]}
            return _coords(face_idx[int(face_id)])
        if n == 6:   # C3D6
            face_idx = {1:[0,1,2], 2:[3,4,5], 3:[0,1,4,3], 4:[1,2,5,4], 5:[2,0,3,5]}
            return _coords(face_idx[int(face_id)])
        if n == 15:  # C3D15
            face_idx = {1:[0,1,2, 6,7,8], 2:[3,4,5, 12,13,14],
                        3:[0,1,4,3, 6,10,12,9], 4:[1,2,5,4, 7,11,13,10],
                        5:[2,0,3,5, 8,9,14,11]}
            return _coords(face_idx[int(face_id)])
        raise NotImplementedError(f"get_face_nodes: element with {n} nodes not supported.")

    def finalize(self) -> None:
        self.nodes.clear()
        for pm in self.parts.values():
            for nid in pm.node_ids:
                self.nodes[nid] = pm.nodes_xyz[nid]
        self.elements.clear()
        for pm in self.parts.values():
            for blk in pm.element_blocks:
                for eid, conn in zip(blk.elem_ids, blk.connectivity):
                    self.elements[eid] = conn

    # —— 便捷查询：根据 interaction 名取摩擦系数（没定义则返回 None）——
    def get_friction_mu(self, interaction_name: Optional[str]) -> Optional[float]:
        if not interaction_name:
            return None
        key = self._dequote(interaction_name)
        if key in self.interactions:
            return self.interactions[key].friction_mu
        # 宽松别名匹配
        for k in self._aliases(key):
            if k in self.interactions:
                return self.interactions[k].friction_mu
        return None

# ======================= 解析器实现 =======================
RE_KW = re.compile(r"^\s*\*([A-Za-z0-9 _-]+)(?:,|$)")
RE_PART = re.compile(r"^\s*\*Part\s*,\s*name\s*=\s*([^,]+)", re.IGNORECASE)
RE_END_PART = re.compile(r"^\s*\*End\s*Part", re.IGNORECASE)
RE_NODE = re.compile(r"^\s*\*Node\b", re.IGNORECASE)
RE_ELEM = re.compile(r"^\s*\*Element\b(.*)$", re.IGNORECASE)
RE_SURFACE = re.compile(r"^\s*\*Surface\b(.*)$", re.IGNORECASE)
RE_TIE = re.compile(r"^\s*\*Tie\b(.*)$", re.IGNORECASE)
RE_CONTACT_PAIR_HEAD = re.compile(r"^\s*\*Contact\s*Pair\b(.*)$", re.IGNORECASE)
RE_ASSEMBLY = re.compile(r"^\s*\*Assembly\b", re.IGNORECASE)
RE_END_ASSEMBLY = re.compile(r"^\s*\*End\s*Assembly\b", re.IGNORECASE)
RE_INSTANCE = re.compile(r"^\s*\*Instance\s*,\s*name\s*=\s*([^,]+)\s*,\s*part\s*=\s*([^,\s]+)", re.IGNORECASE)
RE_BOUNDARY = re.compile(r"^\s*\*Boundary\b", re.IGNORECASE)
RE_CLOAD = re.compile(r"^\s*\*Cload\b", re.IGNORECASE)

# —— sets：支持 nset=/elset=/name= ——
RE_NSET_HEAD  = re.compile(r"^\s*\*Nset\b(.*)$",  re.IGNORECASE)
RE_ELSET_HEAD = re.compile(r"^\s*\*Elset\b(.*)$", re.IGNORECASE)

# —— NEW：接触性质相关 ——
RE_SURFACE_INTERACTION = re.compile(r"^\s*\*(?:Surface\s+Interaction|Contact\s+Property)\b(.*)$", re.IGNORECASE)
RE_FRICTION = re.compile(r"^\s*\*Friction\b(.*)$", re.IGNORECASE)

def _is_comment_or_empty(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("**")

def _extract_param(arg_str: str, key: str) -> Optional[str]:
    if not arg_str: return None
    m = re.search(rf"{key}\s*=\s*([^,]+)", arg_str, re.IGNORECASE)
    return m.group(1).strip() if m else None

def _parse_kw_params(arg_str: str) -> Tuple[Dict[str, str], List[str]]:
    params: Dict[str, str] = {}; flags: List[str] = []
    if not arg_str: return params, flags
    for token in arg_str.split(","):
        tok = token.strip()
        if not tok: continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            params[k.strip().lower()] = v.strip()
        else:
            flags.append(tok.strip().lower())
    return params, flags

def _collect_set_items(lines: List[str], start: int, n: int) -> Tuple[List[str], List[str]]:
    items: List[str] = []; raw_lines: List[str] = []; j = start
    while j < n and not RE_KW.match(lines[j]):
        raw = lines[j].strip()
        if not _is_comment_or_empty(raw):
            raw_lines.append(raw)
            items.extend([p.strip() for p in raw.split(",") if p.strip()])
        j += 1
    return items, raw_lines

def _normalize_surface_items(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    norm: List[Tuple[str, str]] = []
    for name, face in items:
        raw = str(name).strip()
        deq = raw.strip('"').strip("'")
        m = re.match(r'^(.*?)[_\.\-:]S(\d+)$', deq, re.IGNORECASE)
        if m:
            base = m.group(1).strip()
            suf  = f"S{m.group(2)}".upper()
            face2 = face.upper() if (face and str(face).upper().startswith("S")) else suf
            name2 = f'"{base}"' if raw.startswith('"') or raw.startswith("'") else base
            norm.append((name2, face2))
        else:
            norm.append((name, face.upper() if face else face))
    return norm

def load_inp(path: str) -> AssemblyModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"INP not found: {path}")

    model = AssemblyModel()
    current_part: Optional[PartMesh] = None
    in_nodes = False; in_elems = False
    current_elem_type: Optional[str] = None
    current_elem_params: Dict[str, str] = {}
    current_elem_ids: List[int] = []; current_elem_conn: List[List[int]] = []

    scope = "global"; in_assembly = False; current_owner: Optional[str] = None

    # 跟踪最近一次 *Surface Interaction 的名字，供 *Friction 接续解析
    last_interaction_name: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f]

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]; i += 1
        if _is_comment_or_empty(line): continue
        m_kw = RE_KW.match(line)
        if m_kw:
            # 结束上一块 *Element
            if in_elems and current_part and current_elem_type:
                current_part.element_blocks.append(
                    ElementBlock(current_elem_type, current_elem_ids, current_elem_conn, current_elem_params.copy())
                )
                in_elems = False; current_elem_type = None
                current_elem_params = {}; current_elem_ids = []; current_elem_conn = []

            # —— 关键字分派 ——
            if RE_PART.match(line):
                pname = RE_PART.match(line).group(1).strip()
                current_part = PartMesh(name=pname); model.parts[pname] = current_part
                scope, current_owner = "part", pname
                in_nodes = in_elems = False; continue

            if RE_END_PART.match(line):
                current_part = None; scope, current_owner = "global", None
                in_nodes = in_elems = False; continue

            if RE_ASSEMBLY.match(line):
                in_assembly = True; scope, current_owner = "assembly", None; continue

            if RE_END_ASSEMBLY.match(line):
                in_assembly = False; scope, current_owner = "global", None; continue

            if RE_INSTANCE.match(line):
                m = RE_INSTANCE.match(line)
                model.instances.append(InstanceDef(instance=m.group(1).strip(), part=m.group(2).strip()))
                continue

            if RE_NODE.match(line):
                in_nodes, in_elems = True, False; continue

            if RE_ELEM.match(line):
                in_nodes, in_elems = False, True
                arg_str = RE_ELEM.match(line).group(1) or ""
                current_elem_type = _extract_param(arg_str, "type")
                current_elem_params = {}
                for token in arg_str.split(","):
                    tok = token.strip()
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        current_elem_params[k.strip().lower()] = v.strip()
                continue

            if RE_SURFACE.match(line):
                arg_str = RE_SURFACE.match(line).group(1) or ""
                sname = _extract_param(arg_str, "name") or f"surface_at_{i}"
                stype = _extract_param(arg_str, "type") or "UNKNOWN"
                raw_lines: List[str] = []; items: List[Tuple[str, str]] = []
                j = i
                while j < n and not RE_KW.match(lines[j]) and not _is_comment_or_empty(lines[j]):
                    raw = lines[j].strip(); raw_lines.append(raw)
                    parts = [p.strip() for p in raw.split(",") if p.strip()]
                    if len(parts) == 2 and parts[1].upper().startswith("S"):
                        items.append((parts[0], parts[1].upper()))
                    elif len(parts) == 1:
                        items.append((parts[0], ""))
                    else:
                        items.append((parts[0], parts[1] if len(parts) > 1 else ""))
                    j += 1
                items = _normalize_surface_items(items)
                key = sname
                if scope == "part" and current_owner:
                    key = f"{current_owner}::{sname}"
                elif scope == "assembly":
                    key = f"ASM::{sname}"
                model.surfaces[key] = SurfSurfaceDef(
                    stype=stype.upper(), name=sname, items=items,
                    owner=current_owner, scope=("part" if scope=="part" else "assembly"),
                    raw_lines=raw_lines
                )
                i = j; continue

            # —— NEW: Surface Interaction / Contact Property ——
            m_int = RE_SURFACE_INTERACTION.match(line)
            if m_int:
                arg_str = m_int.group(1) or ""
                iname = _extract_param(arg_str, "name") or f"INTERACTION_{len(model.interactions)+1}"
                key = AssemblyModel._dequote(iname)
                model.interactions.setdefault(key, InteractionProp(name=key))
                last_interaction_name = key
                # 不直接前瞻 *Friction；Friction 作为单独关键字出现，下面再处理
                continue

            # —— NEW: Friction ——
            m_fri = RE_FRICTION.match(line)
            if m_fri:
                # 读取下一条非空、非注释、非关键字的数据行，取首个数作为 μ
                j = i
                mu_val: Optional[float] = None
                while j < n and not RE_KW.match(lines[j]):
                    raw = lines[j].strip()
                    if _is_comment_or_empty(raw):
                        j += 1; continue
                    # 解析首个数字
                    toks = [t.strip() for t in raw.split(",") if t.strip()]
                    for t in toks:
                        try:
                            mu_val = float(t)
                            break
                        except Exception:
                            pass
                    if mu_val is not None:
                        break
                    j += 1
                # 记入最近的 interaction
                if last_interaction_name:
                    ip = model.interactions.setdefault(last_interaction_name, InteractionProp(name=last_interaction_name))
                    ip.friction_mu = mu_val
                    if j < n:
                        ip.raw_blocks.append(lines[j].strip() if mu_val is not None else "")
                i = j
                continue

            # —— Contact Pair（带头部参数解析，支持 interaction=xxx）——
            m_cph = RE_CONTACT_PAIR_HEAD.match(line)
            if m_cph:
                head_arg = m_cph.group(1) or ""
                params, _flags = _parse_kw_params(head_arg)
                default_inter = params.get("interaction")
                j = i
                while j < n and not RE_KW.match(lines[j]):
                    raw = lines[j].strip()
                    if not _is_comment_or_empty(raw):
                        parts = [p.strip() for p in raw.split(",") if p.strip()]
                        master = parts[0] if len(parts)>0 else ""
                        slave  = parts[1] if len(parts)>1 else ""
                        # 行内也可能写 interaction=...
                        inter = default_inter
                        for tok in parts[2:]:
                            if "interaction" in tok.lower():
                                inter = tok.split("=")[-1].strip()
                        model.contact_pairs.append(ContactPair(master, slave, inter, raw))
                    j += 1
                i = j; continue

            if RE_TIE.match(line):
                j = i
                while j < n and not RE_KW.match(lines[j]):
                    raw = lines[j].strip()
                    if not _is_comment_or_empty(raw):
                        parts = [p.strip() for p in raw.split(",") if p.strip()]
                        master = parts[0] if len(parts)>0 else ""
                        slave  = parts[1] if len(parts)>1 else ""
                        model.ties.append(TieConstraint(master, slave, raw))
                    j += 1
                i = j; continue

            if RE_BOUNDARY.match(line):
                j = i
                while j < n and not RE_KW.match(lines[j]):
                    raw = lines[j].strip()
                    if not _is_comment_or_empty(raw):
                        model.boundaries.append(BoundaryEntry(raw))
                    j += 1
                i = j; continue

            if RE_CLOAD.match(line):
                j = i
                while j < n and not RE_KW.match(lines[j]):
                    raw = lines[j].strip()
                    if not _is_comment_or_empty(raw):
                        model.loads.append(LoadEntry(raw))
                    j += 1
                i = j; continue

            # 关键：NSET/ELSET 使用“头部匹配 + 参数解析”，支持 nset=/elset=/name=
            m_nh = RE_NSET_HEAD.match(line)
            if m_nh:
                arg_str = m_nh.group(1) or ""
                params, flags = _parse_kw_params(arg_str)
                name = params.get("nset") or params.get("name") or params.get("set")
                items, raw_lines = _collect_set_items(lines, i, n)
                scope_key = "part" if scope=="part" else "assembly"
                owner = current_owner if scope=="part" else None
                model.nsets[name] = SetDef(name, "nset", scope_key, owner, items, raw_lines, params, flags)
                for alias in AssemblyModel._aliases(name):
                    model.nsets.setdefault(alias, model.nsets[name])
                i += len(raw_lines); continue

            m_eh = RE_ELSET_HEAD.match(line)
            if m_eh:
                arg_str = m_eh.group(1) or ""
                params, flags = _parse_kw_params(arg_str)
                name = params.get("elset") or params.get("name") or params.get("set")
                items, raw_lines = _collect_set_items(lines, i, n)
                scope_key = "part" if scope=="part" else "assembly"
                owner = current_owner if scope=="part" else None
                model.elsets[name] = SetDef(name, "elset", scope_key, owner, items, raw_lines, params, flags)
                for alias in AssemblyModel._aliases(name):
                    model.elsets.setdefault(alias, model.elsets[name])
                base = AssemblyModel._strip_suffix_S(name)
                if base and base != name:
                    for alias in AssemblyModel._aliases(base):
                        model.elsets.setdefault(alias, model.elsets[name])
                i += len(raw_lines); continue

            in_nodes = False
            continue

        # ---- 非关键字的数据行 ----
        if in_nodes:
            if not current_part:
                pname = "_GLOBAL_"
                current_part = model.parts.setdefault(pname, PartMesh(name=pname))
                scope, current_owner = "part", pname
            try:
                toks = [t.strip() for t in line.split(",") if t.strip()]
                if len(toks) >= 4:
                    nid = int(float(toks[0])); x, y, z = map(float, toks[1:4])
                    current_part.node_ids.append(nid)
                    current_part.nodes_xyz[nid] = (x, y, z)
            except Exception:
                pass
            continue

        if in_elems and current_part and current_elem_type:
            try:
                toks = [t.strip() for t in line.split(",") if t.strip()]
                if len(toks) >= 2:
                    eid = int(float(toks[0]))
                    conn = [int(float(t)) for t in toks[1:]]
                    current_elem_ids.append(eid)
                    current_elem_conn.append(conn)
            except Exception:
                pass
            continue

    # 收尾最后 *Element 块
    if in_elems and current_part and current_elem_type:
        current_part.element_blocks.append(
            ElementBlock(current_elem_type, current_elem_ids, current_elem_conn, current_elem_params.copy())
        )

    model.finalize()
    return model

# ======================= CLI =======================
def _print_quick_summary(model: AssemblyModel, max_surface: int = 30) -> None:
    print("\n=== INP Quick Summary ===")
    s = model.summary()
    for k, v in s.items():
        if k == "surface_names":
            names = v; more = " ..." if s["num_surfaces"] > len(names) else ""
            print(f"{k}: {names}{more}")
        else:
            print(f"{k}: {v}")
    print("\n--- Parts ---")
    for pname, pm in model.parts.items():
        etypes = list({blk.elem_type for blk in pm.element_blocks})
        print(f"Part '{pname}': nodes={len(pm.node_ids)}, elem_blocks={len(pm.element_blocks)}, types={etypes}")
    print("\n--- Contact Pairs (first 10) ---")
    for cp in model.contact_pairs[:10]:
        mu = model.get_friction_mu(cp.interaction)
        print(f"master='{cp.master}', slave='{cp.slave}', interaction={cp.interaction}, mu={mu}")
    if model.ties:
        print("\n--- Ties ---")
        for t in model.ties[:10]:
            print(f"master='{t.master}', slave='{t.slave}'")
    if model.boundaries:
        print("\n--- Boundary (first 10) ---")
        for b in model.boundaries[:10]:
            print(b.raw)
    if model.interactions:
        print("\n--- Interactions ---")
        for name, ip in model.interactions.items():
            print(f"name='{name}', friction_mu={ip.friction_mu}")
    print("\n--- Nsets / Elsets ---")
    print(f"Nsets: {len(model.nsets)} ; Elsets: {len(model.elsets)}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parse Abaqus .inp")
    ap.add_argument("--inp", type=str, required=True)
    ap.add_argument("--dump_json", type=str, default="")
    args = ap.parse_args()
    mdl = load_inp(args.inp)
    _print_quick_summary(mdl)
    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(mdl.summary(), f, ensure_ascii=False, indent=2)
        print(f"Summary JSON saved to: {args.dump_json}")

if __name__ == "__main__":
    main()
