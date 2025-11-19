bl_info = { 
    "name": "Three d Cursor Snap",
    "author": "Deepak",
    "version": (1, 2, 0),
    "blender": (4, 5, 0),
    "description": "3d cursor snap to vertex/edge/face (with visibility).",
    "category": "3D View",
}

import bpy
import bpy_extras
from mathutils import Vector

VERTEX_RADIUS = 20
EDGE_RADIUS   = 25
FACE_RADIUS   = 20


def sdist(a, b):
    return (Vector(a) - Vector(b)).length


# ===========================================================
#               VISIBILITY CHECK (VERTEX)
# ===========================================================
def is_vertex_visible(context, vertex_world):

    deps = context.evaluated_depsgraph_get()
    rv3d = context.region_data

    cam_origin = rv3d.view_matrix.inverted().translation
    direction = vertex_world - cam_origin
    dist_to_vertex = direction.length

    if dist_to_vertex == 0:
        return True

    direction.normalize()

    hit, hit_loc, *_ = context.scene.ray_cast(deps, cam_origin, direction)
    if hit:
        dist_hit = (hit_loc - cam_origin).length
        if dist_hit < dist_to_vertex - 0.003:
            return False
        if abs(dist_hit - dist_to_vertex) <= 0.01:
            return True

    offset_origin = cam_origin + direction * 0.02
    hit2, hit_loc2, *_ = context.scene.ray_cast(deps, offset_origin, direction)
    if hit2:
        dist_hit2 = (hit_loc2 - offset_origin).length
        dist_vertex2 = (vertex_world - offset_origin).length
        if dist_hit2 < dist_vertex2 - 0.003:
            return False

    return True


# ===========================================================
#       VISIBILITY FOR ANY POINT (EDGE SAMPLE SUPPORT)
# ===========================================================
def is_point_visible(context, point_world):
    deps = context.evaluated_depsgraph_get()
    rv3d = context.region_data

    cam_origin = rv3d.view_matrix.inverted().translation
    direction = point_world - cam_origin
    dist_to_point = direction.length

    if dist_to_point == 0:
        return True

    direction.normalize()

    hit, hit_loc, *_ = context.scene.ray_cast(deps, cam_origin, direction)
    if hit:
        dist_hit = (hit_loc - cam_origin).length
        if dist_hit < dist_to_point - 0.003:
            return False
        if abs(dist_hit - dist_to_point) <= 0.01:
            return True

    offset_origin = cam_origin + direction * 0.02
    hit2, hit_loc2, *_ = context.scene.ray_cast(deps, offset_origin, direction)
    if hit2:
        dist_hit2 = (hit_loc2 - offset_origin).length
        dist_point2 = (point_world - offset_origin).length
        if dist_hit2 < dist_point2 - 0.003:
            return False

    return True


# ===========================================================
#                FIND NEAREST VISIBLE VERTEX
# ===========================================================
def find_nearest_visible_vertex(context, mouse):

    region = context.region
    rv3d = context.region_data

    best = (None, 999999)

    for obj in context.view_layer.objects:
        if not obj.visible_get():
            continue
        if obj.type != "MESH":
            continue

        for v in obj.data.vertices:
            pw = obj.matrix_world @ v.co

            if not is_vertex_visible(context, pw):
                continue

            ps = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, pw)
            if ps is None:
                continue

            d = sdist(ps, mouse)
            if d < VERTEX_RADIUS and d < best[1]:
                best = (pw, d)

    return best[0]


# ===========================================================
#                         RAYCAST
# ===========================================================
def evaluated_raycast(context, mouse):

    region = context.region
    rv3d   = context.region_data
    deps   = context.evaluated_depsgraph_get()

    view = bpy_extras.view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    origin = bpy_extras.view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)

    hit, loc, normal, fi, obj, _ = context.scene.ray_cast(deps, origin, view)

    if hit:
        return True, loc, fi, obj
    return False, None, None, None


# ===========================================================
#  FINAL — EDGE MIDPOINT SNAP (with EDIT MODE FIX)
# ===========================================================
def edge_face_mid_snap(context, mouse):

    region = context.region
    rv3d   = context.region_data
    deps   = context.evaluated_depsgraph_get()

    ok, hit_loc, face_idx, obj = evaluated_raycast(context, mouse)
    if not ok or obj.type != "MESH":
        return None

    obj_eval = obj.evaluated_get(deps)

    # ------------------------------
    # FIX: edit mode uses obj.data
    # ------------------------------
    if obj.mode == 'EDIT':
        mesh = obj.data
    else:
        mesh = obj_eval.data

    if face_idx < 0 or face_idx >= len(mesh.polygons):
        return hit_loc

    poly = mesh.polygons[face_idx]
    loop_indices = list(poly.loop_indices)
    verts = poly.vertices

    w = obj.matrix_world
    mouse_vec = Vector(mouse)

    best = (None, None, 999999)
    sample_factors = (0.25, 0.5, 0.75)

    for i, loop_idx in enumerate(loop_indices):

        v1 = w @ mesh.vertices[mesh.loops[loop_idx].vertex_index].co
        next_loop = loop_indices[(i + 1) % len(loop_indices)]
        v2 = w @ mesh.vertices[mesh.loops[next_loop].vertex_index].co

        midpoint = (v1 + v2) * 0.5   # always snap to this

        for f in sample_factors:
            sample = v1.lerp(v2, f)

            if not is_point_visible(context, sample):
                continue

            ps = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, sample)
            if ps is None:
                continue

            d = sdist(ps, mouse_vec)

            if d < best[2]:
                best = (midpoint, sample, d)

    if best[0] and best[2] <= EDGE_RADIUS:
        return best[0]   # midpoint always

    # ---------- FACE CENTER ----------
    fc = Vector((0,0,0))
    for vid in verts:
        fc += mesh.vertices[vid].co
    fc /= len(verts)

    fc_world = w @ fc

    fs = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, fc_world)
    if fs and sdist(fs, mouse_vec) <= FACE_RADIUS:
        return fc_world

    return None


# ===========================================================
#                        CURVE SNAP
# ===========================================================
def curve_snap_points(obj, deps):
    pts = []
    w = obj.matrix_world

    if obj.type != "CURVE":
        return pts

    crv = obj.data

    for sp in crv.splines:
        if sp.type == 'BEZIER':
            for p in sp.bezier_points:
                pts.append(w @ p.co)
                pts.append(w @ p.handle_left)
                pts.append(w @ p.handle_right)
        else:
            for p in sp.points:
                pts.append(w @ p.co.xyz)

    crv_eval = obj.evaluated_get(deps).data
    for sp in crv_eval.splines:
        for p in sp.points:
            pts.append(w @ p.co.xyz)

    return pts


def curve_snap(context, mouse):

    region = context.region
    rv3d   = context.region_data
    deps   = context.evaluated_depsgraph_get()

    for obj in context.view_layer.objects:
        if obj.type != "CURVE" or not obj.visible_get():
            continue

        for p in curve_snap_points(obj, deps):
            ps = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, p)
            if ps and sdist(ps, mouse) <= VERTEX_RADIUS:
                return p

    return None


# ===========================================================
#                        FREE SPACE
# ===========================================================
def free_space_point(context, mouse):

    region = context.region
    rv3d   = context.region_data
    deps   = context.evaluated_depsgraph_get()

    view = bpy_extras.view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse)
    origin = bpy_extras.view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse)

    hit, loc, *_ = context.scene.ray_cast(deps, origin, view)
    if hit:
        return loc

    return origin + view * 50.0


# ===========================================================
#                        MASTER SNAP
# ===========================================================
def snap_point(context, mouse):

    v = find_nearest_visible_vertex(context, mouse)
    if v:
        return v

    ef = edge_face_mid_snap(context, mouse)
    if ef:
        return ef

    c = curve_snap(context, mouse)
    if c:
        return c

    return free_space_point(context, mouse)


def place_cursor(context, mouse):
    context.scene.cursor.location = free_space_point(context, mouse)


# ===========================================================
#                        OPERATOR
# ===========================================================
class CURSOR_OT_snap_drag(bpy.types.Operator):
    bl_idname = "view3d.cursor_snap_drag"
    bl_label  = "Cursor Snap Drag 4.5"
    bl_options = {'BLOCKING'}

    dragging = False
    prev_wire = None
    prev_opacity = None

    def modal(self, context, event):

        if event.type == "RIGHTMOUSE" and event.value == "RELEASE":

            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    overlay = area.spaces.active.overlay
                    overlay.show_wireframes = self.prev_wire
                    overlay.wireframe_opacity = self.prev_opacity

            if not self.dragging:
                place_cursor(context, (event.mouse_region_x, event.mouse_region_y))

            return {'FINISHED'}

        if event.type == "MOUSEMOVE":
            self.dragging = True
            p = snap_point(context, (event.mouse_region_x, event.mouse_region_y))
            if p:
                context.scene.cursor.location = p

        if event.type == "ESC":
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    overlay = area.spaces.active.overlay
                    overlay.show_wireframes = self.prev_wire
                    overlay.wireframe_opacity = self.prev_opacity

            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):

        if event.shift and event.type == "RIGHTMOUSE" and event.value == "PRESS":

            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    overlay = area.spaces.active.overlay
                    self.prev_wire = overlay.show_wireframes
                    self.prev_opacity = overlay.wireframe_opacity

                    overlay.show_wireframes = True
                    overlay.wireframe_opacity = 1.0

            self.dragging = False
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}

        return {'CANCELLED'}


# ===========================================================
#                        REGISTER
# ===========================================================
addon_keymaps = []

def register():
    bpy.utils.register_class(CURSOR_OT_snap_drag)

    wm = bpy.context.window_manager
    if not wm.keyconfigs.addon:
        return

    km = wm.keyconfigs.addon.keymaps.new(
        name="3D View", space_type="VIEW_3D")

    kmi = km.keymap_items.new(
        "view3d.cursor_snap_drag",
        "RIGHTMOUSE", "PRESS", shift=True)

    addon_keymaps.append((km, kmi))


def unregister():
    bpy.utils.unregister_class(CURSOR_OT_snap_drag)

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)

    addon_keymaps.clear()


if __name__ == "__main__":
    register()
