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
import bpy, glob, os, sys, math
import numpy as np
import mathutils

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

def _add_atomic(npz, coll, radius=1.6):
    """Cα backbone tubes: hexamers gray, pentamers red (from precompute_backbone.py)."""
    d = np.load(npz); coords = d['coords']; lengths = d['lengths']; types = d['types']
    for target, cname, rgba in [(0, "capsomers_hex", (0.62, 0.64, 0.68, 1)), (1, "capsomers_pent", (0.93, 0.11, 0.11, 1))]:
        cu = bpy.data.curves.new(cname, 'CURVE'); cu.dimensions = '3D'
        cu.bevel_depth = radius; cu.bevel_resolution = 1; cu.use_fill_caps = True
        i = 0
        for L, ty in zip(lengths, types):
            L = int(L); seg = coords[i:i + L]; i += L
            if int(ty) != target: continue
            sp = cu.splines.new('POLY'); sp.points.add(L - 1)
            sp.points.foreach_set("co", np.concatenate([seg, np.ones((L, 1), 'f4')], axis=1).ravel().tolist())
        cu.materials.append(_mat(cname, rgba))
        coll.objects.link(bpy.data.objects.new(cname, cu))

def load_set(name, atomic=True):
    """Load one mesh's surface + cage + curvature points (+ atomic capsid) into '<name>_set'."""
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
    bb = glob.glob(f"{DIR}/{name}_C*_backbone.npz")
    if atomic and bb: _add_atomic(bb[0], coll)
    print("  loaded %s_set" % name); return name

def load_all():
    return [n for n in NAMES if load_set(n)]

def build_movie(fps=24, sec_per_turn=10.0, gap=280.0, resx=1280, resy=720,
                out="/Users/olson/Dev/fullerenes/series_movie", do_render=False):
    """Scene 'series_movie': complexes arrayed along X, long axes along Y, each
    spinning about its own long (Y) axis, sec_per_turn per turn (continuous).
    3-turn buildup: turn 1 = surface + curvature points; turn 2 adds the fullerene
    cage; turn 3 adds the atomic capsid. Orthographic camera along Z. Total length
    = 3*sec_per_turn."""
    scn = bpy.data.scenes.get("series_movie") or bpy.data.scenes.new("series_movie")
    bpy.context.window.scene = scn
    for o in list(scn.collection.objects): bpy.data.objects.remove(o, do_unlink=True)
    coll = scn.collection
    try: bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'   # constant-speed spin
    except Exception: pass
    comps = []
    for name in NAMES:
        sg = glob.glob(f"{DIR}/{name}*_surface.obj")
        if not sg: continue
        V, F = _readobj(sg[0]); Va = np.array(V); c = Va.mean(0)
        _, _, vt = np.linalg.svd(Va - c, full_matrices=False)
        R = np.array([vt[1], vt[0], vt[2]])              # long axis (vt0) -> Y
        if np.linalg.det(R) < 0: R[0] = -R[0]            # keep proper (no mirror)
        Vc = (Va - c) @ R.T
        comps.append(dict(name=name, c=c, R=R,
                          rad=float(np.sqrt(Vc[:, 0] ** 2 + Vc[:, 2] ** 2).max()),
                          halfY=float(np.abs(Vc[:, 1]).max())))
    N = len(comps); spacing = 2 * max(x['rad'] for x in comps) + gap; X0 = -(N - 1) / 2.0 * spacing
    tf = int(round(fps * sec_per_turn)); total = 3 * tf          # frames per turn; total (3 turns)
    cage_af, atom_af = tf + 1, 2 * tf + 1                        # cage appears at turn 2, atomic at turn 3
    def canon(P, c, R): return (np.asarray(P, float) - c) @ R.T
    def parent(o, emp): o.parent = emp; o.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    def appear(o, af):                                           # keyframe object to pop in at frame af
        o.hide_viewport = True; o.hide_render = True
        o.keyframe_insert("hide_viewport", frame=1); o.keyframe_insert("hide_render", frame=1)
        o.hide_viewport = False; o.hide_render = False
        o.keyframe_insert("hide_viewport", frame=af); o.keyframe_insert("hide_render", frame=af)
    def mkmesh(nm, Vc, F, material, emp, smooth=True, wire=False):
        me = bpy.data.meshes.new(nm); me.from_pydata([tuple(v) for v in Vc], [], [list(map(int, f)) for f in F]); me.update()
        if smooth:
            for pg in me.polygons: pg.use_smooth = True
        me.materials.append(material); o = bpy.data.objects.new(nm, me); coll.objects.link(o)
        if wire: o.modifiers.new("w", "WIREFRAME").thickness = 3.0
        parent(o, emp); return o
    for i, cx in enumerate(comps):
        name, c, R = cx['name'], cx['c'], cx['R']
        emp = bpy.data.objects.new("cx_" + name, None); coll.objects.link(emp); emp.location = (X0 + i * spacing, 0, 0)
        V, F = _readobj(glob.glob(f"{DIR}/{name}*_surface.obj")[0]); mkmesh("surf_" + name, canon(V, c, R), F, _mat("MV_surf", (0.5, 0.68, 0.95, 1), alpha=0.10, blend=True), emp)
        V, F = _readobj(glob.glob(f"{DIR}/{name}_C*_cage.obj")[0]); cg = mkmesh("cage_" + name, canon(V, c, R), F, _mat("MV_cage", (0.85, 0.55, 0.22, 1)), emp, wire=True); appear(cg, cage_af)
        V, F = _readobj(glob.glob(f"{DIR}/{name}_C*_curvature.obj")[0]); mkmesh("curv_" + name, canon(V, c, R), F, _mat("MV_curv", (0.10, 0.85, 0.20, 1), emit=0.8), emp, smooth=False)
        bb = glob.glob(f"{DIR}/{name}_C*_backbone.npz")
        if bb:
            d = np.load(bb[0]); co = canon(d['coords'], c, R).astype('f4'); lengths = d['lengths']; types = d['types']
            for tgt, cn, rgba in [(0, "hex_" + name, (0.62, 0.64, 0.68, 1)), (1, "pent_" + name, (0.93, 0.11, 0.11, 1))]:
                cu = bpy.data.curves.new(cn, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = 1.6; cu.bevel_resolution = 0; cu.use_fill_caps = True
                j = 0
                for L, ty in zip(lengths, types):
                    L = int(L); seg = co[j:j + L]; j += L
                    if int(ty) != tgt: continue
                    sp = cu.splines.new('POLY'); sp.points.add(L - 1)
                    sp.points.foreach_set("co", np.concatenate([seg, np.ones((L, 1), 'f4')], axis=1).ravel().tolist())
                cu.materials.append(_mat("MV_" + cn.split("_")[0], rgba)); o = bpy.data.objects.new(cn, cu); coll.objects.link(o); parent(o, emp); appear(o, atom_af)
        emp.rotation_euler = (0, 0, 0); emp.keyframe_insert("rotation_euler", index=1, frame=1)
        emp.rotation_euler = (0, 2 * math.pi * 3, 0); emp.keyframe_insert("rotation_euler", index=1, frame=total + 1)
    cam = bpy.data.cameras.new("moviecam"); cam.type = 'ORTHO'; cam.ortho_scale = N * spacing * 1.08; cam.clip_end = 40000
    co = bpy.data.objects.new("moviecam", cam); coll.objects.link(co); co.location = (0, 0, 8000); co.rotation_euler = (0, 0, 0); scn.camera = co
    sun = bpy.data.lights.new("sun", "SUN"); sun.energy = 3.5; so = bpy.data.objects.new("sun", sun); coll.objects.link(so); so.rotation_euler = (0.5, 0.2, 0)
    if scn.world is None: scn.world = bpy.data.worlds.new("W")
    try:
        bg = scn.world.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.03, 0.03, 0.05, 1); bg.inputs[1].default_value = 1.0
    except Exception: pass
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try: scn.render.engine = eng; break
        except Exception: pass
    try: scn.eevee.taa_render_samples = 16
    except Exception: pass
    scn.render.resolution_x = resx; scn.render.resolution_y = resy; scn.render.fps = fps
    scn.frame_start = 1; scn.frame_end = total
    try:
        scn.render.image_settings.file_format = 'FFMPEG'; scn.render.ffmpeg.format = 'MPEG4'; scn.render.ffmpeg.codec = 'H264'; scn.render.filepath = out + ".mp4"
    except Exception:                                       # Blender built without ffmpeg -> PNG sequence
        os.makedirs("/tmp/series_frames", exist_ok=True)
        scn.render.image_settings.file_format = 'PNG'; scn.render.filepath = "/tmp/series_frames/frame_"
    print("series_movie: %d complexes, spacing %.0f A, 3 turns x %.0fs = frames 1-%d @ %d fps ; cage@f%d atomic@f%d" % (N, spacing, sec_per_turn, total, fps, cage_af, atom_af))
    if do_render: bpy.ops.render.render(animation=True)
    return N

if __name__ == "__main__":
    arg = "all"
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        if rest: arg = rest[0]
    if arg == "all": load_all()
    else: load_set(arg)
