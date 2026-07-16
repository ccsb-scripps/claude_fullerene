#!/usr/bin/env python3
"""Load a mesh's co-registered display set into Blender, one scene per mesh
('<name>_set'):
    - surface          : the cryo-ET segmentation surface (translucent blue)
    - fullerene_cage   : the fullerene cage, shown as an edge lattice (Wireframe modifier)
    - curvature_points : the 12 mesh curvature high points (bowl-peak / pentamer
                         sites) as bright-green octahedral markers
All three (plus the atomic capsid, if you expand it) share one Angstrom frame.

Files are read from capsid_models/ next to this script:
    <name>*_surface.obj , <name>_C*_cage.obj , <name>_C*_curvature.obj

Usage
  In Blender (Scripting workspace, Text editor -> Run, or the console):
      exec(open('/Users/olson/Dev/fullerenes/blender_load_set.py').read())   # loads all 7
      load_set('cage_5')                                                     # just one
  Headless:
      blender --python blender_load_set.py -- cage_5        # one mesh
      blender --python blender_load_set.py -- all           # all 7 (default)
"""
import bpy, glob, os, sys

try: HERE = os.path.dirname(os.path.abspath(__file__))
except NameError: HERE = "/Users/olson/Dev/fullerenes"
DIR = os.path.join(HERE, "capsid_models")
NAMES = ["mesh4a", "mesh5", "cage_3", "cage_4", "cage_5", "cage_6", "cage_7"]

def _readobj(fn):
    V, F = [], []
    for ln in open(fn):
        if ln.startswith("v "):
            p = ln.split(); V.append((float(p[1]), float(p[2]), float(p[3])))
        elif ln.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:]])
    return V, F

def _mat(name, rgba, alpha=1.0, blend=False, emit=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = rgba
    b.inputs["Alpha"].default_value = alpha; b.inputs["Roughness"].default_value = 0.5
    if emit:
        try:
            b.inputs["Emission Color"].default_value = rgba
            b.inputs["Emission Strength"].default_value = emit
        except Exception: pass
    if blend: m.blend_method = 'BLEND'
    return m

def _add(name, V, F, material, coll, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V], [], [list(map(int, f)) for f in F]); me.update()
    if smooth:
        for pg in me.polygons: pg.use_smooth = True
    o = bpy.data.objects.new(name, me); coll.objects.link(o); me.materials.append(material)
    return o

def load_set(name):
    """Load one mesh's surface + cage + curvature points into scene '<name>_set'."""
    surf = glob.glob(f"{DIR}/{name}*_surface.obj")
    cage = glob.glob(f"{DIR}/{name}_C*_cage.obj")
    curv = glob.glob(f"{DIR}/{name}_C*_curvature.obj")
    if not (surf and cage and curv):
        print("  %s: missing files, skipped" % name); return None
    scn = bpy.data.scenes.get(name + "_set") or bpy.data.scenes.new(name + "_set")
    bpy.context.window.scene = scn
    for o in list(scn.collection.objects): bpy.data.objects.remove(o, do_unlink=True)
    coll = scn.collection
    V, F = _readobj(surf[0]); _add("surface", V, F, _mat("SET_surf", (0.5, 0.68, 0.95, 1), alpha=0.12, blend=True), coll)
    V, F = _readobj(cage[0]); c = _add("fullerene_cage", V, F, _mat("SET_cage", (0.80, 0.55, 0.25, 1)), coll)
    c.modifiers.new("wire", "WIREFRAME").thickness = 3.0
    V, F = _readobj(curv[0]); _add("curvature_points", V, F, _mat("SET_curv", (0.10, 0.85, 0.20, 1), emit=0.7), coll, smooth=False)
    print("  loaded %s_set" % name); return name

def load_all():
    return [n for n in NAMES if load_set(n)]

if __name__ == "__main__":
    arg = "all"
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        if rest: arg = rest[0]
    if arg == "all": load_all()
    else: load_set(arg)
