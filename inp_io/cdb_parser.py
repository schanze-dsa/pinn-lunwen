#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cdb_parser.py -- Parse ANSYS CDB into AssemblyModel (nodes/elements/components/contact/bcs).

Scope:
  - NBLOCK / EBLOCK / ETBLOCK
  - CMBLOCK element components (contact groups + parts)
  - D constraints (node-based DOF fixes)

Notes:
  - SOLID185 is mapped as 8-node hex.
  - Contact/target elements (CONTA173/TARGE170) are kept so contact surfaces
    can be triangulated later.
  - If MIRROR1/MIRROR2 both exist, they are merged into a single "MIRROR" part.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

from inp_io.inp_parser import (
    AssemblyModel,
    PartMesh,
    ElementBlock,
    ContactPair,
    BoundaryEntry,
)
from assembly.surfaces import SurfaceDef

# ----------------------------- helpers -----------------------------


def _parse_fixed_width(line: str, widths: Iterable[int]) -> List[str]:
    out: List[str] = []
    idx = 0
    line = line.rstrip("\n")
    total = sum(widths)
    if len(line) < total:
        line = line + (" " * (total - len(line)))
    for w in widths:
        out.append(line[idx : idx + w].strip())
        idx += w
    return out


def _safe_int(val: str) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _safe_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _expand_range_stream(values: List[int]) -> List[int]:
    out: List[int] = []
    last = None
    for v in values:
        if v is None:
            continue
        if v < 0 and last is not None:
            out.extend(range(last, abs(v) + 1))
        else:
            out.append(v)
            last = v
    return out


# ----------------------------- parse blocks -----------------------------


def _parse_etblock(lines: List[str], start: int) -> Tuple[Dict[int, int], int]:
    # ETBLOCK, n, n
    etype_map: Dict[int, int] = {}
    i = start + 1  # next line is format
    i += 1
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("-1"):
            i += 1
            break
        # format is (2i9,19a9) -> two ints at width 9
        parts = _parse_fixed_width(lines[i], [9, 9])
        tid = _safe_int(parts[0])
        code = _safe_int(parts[1])
        if tid is not None and code is not None:
            etype_map[tid] = code
        i += 1
    return etype_map, i


def _parse_nblock(lines: List[str], start: int) -> Tuple[Dict[int, Tuple[float, float, float]], int]:
    nodes: Dict[int, Tuple[float, float, float]] = {}
    i = start + 1  # format line
    
    # 动态解析格式行，例如 "(3i8,6e16.9)" 或 "(3i9,6e21.13e3)"
    format_line = lines[i].strip() if i < len(lines) else ""
    int_width = 9   # 默认值
    float_width = 21  # 默认值
    
    # 匹配格式如 (3i8,6e16.9) 或 (3i9,6e21.13e3)
    m = re.match(r"\((\d+)i(\d+),\d+e(\d+)", format_line)
    if m:
        int_width = int(m.group(2))
        float_width = int(m.group(3))
    
    i += 1
    int_widths = [int_width, int_width, int_width]
    float_widths = [float_width] * 6
    
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        # End markers
        if s.startswith("N,") or s.startswith("EBLOCK") or s.startswith("CMBLOCK") or s.startswith("CM,") or s.startswith("-1"):
            break
        head = _parse_fixed_width(lines[i], int_widths)
        nid = _safe_int(head[0])
        if nid is None:
            i += 1
            continue
        tail = _parse_fixed_width(lines[i][sum(int_widths) :], float_widths)
        x = _safe_float(tail[0])
        y = _safe_float(tail[1])
        z = _safe_float(tail[2])
        if x is None or y is None or z is None:
            i += 1
            continue
        nodes[int(nid)] = (float(x), float(y), float(z))
        i += 1
    return nodes, i


def _parse_eblock(
    lines: List[str],
    start: int,
    etype_map: Dict[int, int],
) -> Tuple[Dict[int, Tuple[str, List[int]]], int]:
    elements: Dict[int, Tuple[str, List[int]]] = {}
    i = start + 1  # format line
    
    # 动态解析格式行，例如 "(19i8)" 或 "(19i10)"
    format_line = lines[i].strip() if i < len(lines) else ""
    int_width = 10   # 默认值
    num_fields = 19  # 默认值
    
    # 匹配格式如 (19i8)
    m = re.match(r"\((\d+)i(\d+)\)", format_line)
    if m:
        num_fields = int(m.group(1))
        int_width = int(m.group(2))
    
    i += 1
    widths = [int_width] * num_fields
    
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("-1"):
            i += 1
            break
        if s.startswith("CMBLOCK") or s.startswith("RLBLOCK"):
            break
        fields = _parse_fixed_width(lines[i], widths)
        ints = [_safe_int(x) for x in fields]
        if len(ints) < 11:
            i += 1
            continue
        elem_id = ints[10]
        type_id = ints[1]
        nnode = ints[8] if ints[8] is not None else 0
        if elem_id is None or type_id is None:
            i += 1
            continue
        node_ids: List[int] = []
        if nnode > 0:
            raw_nodes = ints[11 : 11 + nnode]
            node_ids = [int(n) for n in raw_nodes if n is not None and int(n) != 0]
        code = etype_map.get(int(type_id))
        etype = _etype_name_from_code(code)
        # SOLID186 reduced to first 8 nodes (corner nodes) for DFEM/C3D8 handling.
        if etype == "SOLID186" and len(node_ids) >= 8:
            node_ids = node_ids[:8]
            etype = "SOLID185"
        elements[int(elem_id)] = (etype, node_ids)
        i += 1
    return elements, i


def _parse_cmblock(lines: List[str], start: int) -> Tuple[str, str, List[int], int]:
    # CMBLOCK,NAME,ELEM, <n>
    header = lines[start].strip()
    parts = [p.strip() for p in header.split(",")]
    name = parts[1] if len(parts) > 1 else f"CMP_{start}"
    ctype = parts[2] if len(parts) > 2 else "ELEM"
    i = start + 1  # format line
    if i < len(lines) and lines[i].strip().startswith("("):
        i += 1
    data_vals: List[int] = []
    widths = [10] * 8
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("CMBLOCK") or s.startswith("RLBLOCK") or s.startswith("NBLOCK") or s.startswith("EBLOCK"):
            break
        if s.startswith("-1"):
            i += 1
            break
        # parse 8i10
        fields = _parse_fixed_width(lines[i], widths)
        ints = [_safe_int(x) for x in fields]
        for v in ints:
            if v is not None:
                data_vals.append(int(v))
        i += 1
    ids = _expand_range_stream(data_vals)
    return name, ctype, ids, i


def _etype_name_from_code(code: Optional[int]) -> str:
    if code is None:
        return "UNKNOWN"
    if code == 170:
        return "TARGE170"
    if code == 173:
        return "CONTA173"
    if code == 174:
        return "CONTA174"
    if code == 185:
        return "SOLID185"
    if code == 186:
        return "SOLID186"
    if code == 154:
        return "SURF154"
    return f"ET_{code}"


def _is_contact_component(name: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"GROUP_TARG_CONT_(\d+)_(MASTER|SLAVE)_COMP", name, re.IGNORECASE)
    if not m:
        return None
    return m.group(1), m.group(2).upper()


def _is_combined_component(name: str, components: Dict[str, List[int]]) -> bool:
    # Do not auto-skip base components (e.g., LUOMU/LUOSHUAN) since they may be
    # distinct parts in the CDB.
    return False


# ----------------------------- main loader -----------------------------


def load_cdb(path: str) -> AssemblyModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CDB not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    etype_map: Dict[int, int] = {}
    nodes: Dict[int, Tuple[float, float, float]] = {}
    elements: Dict[int, Tuple[str, List[int]]] = {}
    components: Dict[str, List[int]] = {}
    component_types: Dict[str, str] = {}
    boundaries: List[BoundaryEntry] = []
    active_cm_name: Optional[str] = None
    hmname_pending: bool = False

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("ETBLOCK"):
            etype_map, i = _parse_etblock(lines, i)
            continue
        # 支持单独的 ET,tid,code 格式，例如 ET,7,185
        if line.startswith("ET,"):
            parts = line.split(",")
            if len(parts) >= 3:
                tid = _safe_int(parts[1])
                code = _safe_int(parts[2])
                if tid is not None and code is not None:
                    etype_map[int(tid)] = int(code)
            i += 1
            continue
        if line.startswith("NBLOCK"):
            nodes, i = _parse_nblock(lines, i)
            continue
        # HyperMesh exports component names via comment blocks:
        # !!HMNAME COMP
        # !!   <id> "<COMP_NAME>"
        if line.lstrip().startswith("!!HMNAME") and "COMP" in line.upper():
            hmname_pending = True
            i += 1
            continue
        if hmname_pending and line.lstrip().startswith("!!"):
            m = re.search(r'"([^"]+)"', line)
            if m:
                name = m.group(1)
                if name:
                    # HMNAME appears immediately before EBLOCK in HyperMesh exports,
                    # so prefer it over any previously pending CM name.
                    active_cm_name = name
                    components.setdefault(name, [])
                    component_types[name] = "ELEM"
            hmname_pending = False
            i += 1
            continue
        if hmname_pending and not line.lstrip().startswith("!!"):
            hmname_pending = False
        if line.startswith("EBLOCK"):
            elements_block, i = _parse_eblock(lines, i, etype_map)
            elements.update(elements_block)  # 合并多个 EBLOCK
            if active_cm_name:
                components.setdefault(active_cm_name, []).extend(elements_block.keys())
                component_types.setdefault(active_cm_name, "ELEM")
                active_cm_name = None
            continue
        if line.startswith("CMBLOCK"):
            name, ctype, ids, i = _parse_cmblock(lines, i)
            components[name] = ids
            component_types[name] = ctype.upper()
            continue
        if line.startswith("CM,"):
            parts = [p.strip() for p in line.split(",")]
            name = parts[1] if len(parts) > 1 else ""
            ctype = parts[2].upper() if len(parts) > 2 else ""
            if name and ctype == "ELEM":
                active_cm_name = name
                components.setdefault(name, [])
                component_types[name] = "ELEM"
            i += 1
            continue
        if line.startswith("D,"):
            boundaries.append(BoundaryEntry(raw=line.strip()))
            i += 1
            continue
        i += 1

    model = AssemblyModel()
    model.boundaries = boundaries

    # De-duplicate component element ids
    for name, ids in list(components.items()):
        if ids:
            components[name] = sorted(set(int(v) for v in ids))


    # Build contact pairs from component names
    contact_pairs: List[ContactPair] = []
    contact_groups: Dict[str, Dict[str, str]] = {}
    for name in components.keys():
        hit = _is_contact_component(name)
        if not hit:
            continue
        idx, role = hit
        group = contact_groups.setdefault(idx, {})
        group[role] = name
    for idx, group in sorted(contact_groups.items(), key=lambda kv: int(kv[0])):
        master = group.get("MASTER", "")
        slave = group.get("SLAVE", "")
        if master and slave:
            contact_pairs.append(ContactPair(master=master, slave=slave, interaction=None, raw=""))
    model.contact_pairs = contact_pairs

    # Build parts from component sets (skip contact components)
    part_components: Dict[str, List[int]] = {}
    for name, ids in components.items():
        if _is_contact_component(name):
            continue
        if _is_combined_component(name, components):
            continue
        if component_types.get(name, "") != "ELEM":
            continue
        # Skip pure surface components (e.g., SURF154-only) when building parts.
        if ids:
            only_surface = True
            for eid in ids:
                etype, _ = elements.get(int(eid), ("", []))
                et = (etype or "").upper()
                if et.startswith("CONTA") or et.startswith("TARGE"):
                    continue
                if et in {"SURF154", "ET_154"}:
                    continue
                only_surface = False
                break
            if only_surface:
                continue
        part_components[name] = ids

    # Merge mirror parts if both exist
    mirror1 = next((n for n in part_components if n.upper() == "MIRROR1"), None)
    mirror2 = next((n for n in part_components if n.upper() == "MIRROR2"), None)
    if mirror1 and mirror2:
        merged = sorted(set(part_components[mirror1]) | set(part_components[mirror2]))
        part_components["MIRROR"] = merged
        del part_components[mirror1]
        del part_components[mirror2]

    # 如果没有任何部件定义（没有 CMBLOCK），创建一个包含所有实体元素的默认部件
    if not part_components:
        solid_elem_ids = [
            eid for eid, (etype, _) in elements.items()
            if not etype.startswith("CONTA") and not etype.startswith("TARGE")
        ]
        if solid_elem_ids:
            part_components["ALL_SOLID"] = solid_elem_ids
            print(f"[cdb_parser] 未发现部件定义(CMBLOCK)，创建默认部件 ALL_SOLID: {len(solid_elem_ids)} 个元素")

    # Create parts
    for name, elem_ids in part_components.items():
        part = PartMesh(name=name)
        blocks: Dict[str, Tuple[List[int], List[List[int]]]] = {}
        for eid in elem_ids:
            if eid not in elements:
                continue
            etype, conn = elements[eid]
            if etype.startswith("CONTA") or etype.startswith("TARGE"):
                continue
            blk = blocks.setdefault(etype, ([], []))
            blk[0].append(int(eid))
            blk[1].append([int(n) for n in conn])
        for etype, (eids, conns) in blocks.items():
            part.element_blocks.append(ElementBlock(etype, eids, conns, {}))

        # nodes for part
        node_ids = set()
        for blk in part.element_blocks:
            for conn in blk.connectivity:
                for nid in conn:
                    node_ids.add(int(nid))
        part.node_ids = sorted(node_ids)
        part.nodes_xyz = {nid: nodes[nid] for nid in part.node_ids if nid in nodes}
        model.parts[name] = part

    # Contact elements in a separate part for surface triangulation
    contact_elem_ids = [
        eid for eid, (etype, _) in elements.items()
        if etype.startswith("CONTA") or etype.startswith("TARGE")
    ]
    if contact_elem_ids:
        part = PartMesh(name="__CONTACT__")
        blocks: Dict[str, Tuple[List[int], List[List[int]]]] = {}
        for eid in contact_elem_ids:
            etype, conn = elements[eid]
            blk = blocks.setdefault(etype, ([], []))
            blk[0].append(int(eid))
            blk[1].append([int(n) for n in conn])
        for etype, (eids, conns) in blocks.items():
            part.element_blocks.append(ElementBlock(etype, eids, conns, {}))
        node_ids = set()
        for blk in part.element_blocks:
            for conn in blk.connectivity:
                for nid in conn:
                    node_ids.add(int(nid))
        part.node_ids = sorted(node_ids)
        part.nodes_xyz = {nid: nodes[nid] for nid in part.node_ids if nid in nodes}
        model.parts[part.name] = part

    # Surface definitions from components (element sets)
    for name, ids in components.items():
        if component_types.get(name, "") != "ELEM":
            continue
        items = [(int(eid), "") for eid in ids if eid in elements]
        model.surfaces[name] = SurfaceDef(
            stype="ELEMENT",
            name=name,
            items=items,
            owner=None,
            scope="assembly",
            raw_lines=None,
        )

    # Convenience: create a mirror-up alias for visualization when mirror part exists.
    mirror_part = None
    for cand in ("MIRROR", "MIRROR1"):
        if cand in model.parts:
            mirror_part = cand
            break
    if mirror_part and "MIRROR UP" not in model.surfaces:
        # Use one element to seed part resolution; viz can rebuild part_top surface later.
        try:
            part = model.parts[mirror_part]
            seed_eid = None
            for blk in part.element_blocks:
                if blk.elem_ids:
                    seed_eid = int(blk.elem_ids[0])
                    break
            if seed_eid is not None:
                model.surfaces["MIRROR UP"] = SurfaceDef(
                    stype="ELEMENT",
                    name="MIRROR UP",
                    items=[(seed_eid, "S1")],
                    owner=mirror_part,
                    scope="part",
                    raw_lines=None,
                )
        except Exception:
            pass

    # Fill nodes/elements at assembly level
    model.nodes = nodes
    model.elements = {eid: conn for eid, (_, conn) in elements.items()}
    model.element_types = {eid: etype for eid, (etype, _) in elements.items()}
    return model


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Parse ANSYS .cdb")
    ap.add_argument("--cdb", type=str, required=True)
    ap.add_argument("--dump_json", type=str, default="")
    args = ap.parse_args()
    asm = load_cdb(args.cdb)
    print(asm.summary())
    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(asm.summary(), f, ensure_ascii=False, indent=2)
        print(f"Summary JSON saved to: {args.dump_json}")


if __name__ == "__main__":
    main()
