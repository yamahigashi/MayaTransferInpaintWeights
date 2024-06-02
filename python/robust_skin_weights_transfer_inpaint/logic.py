"""
https://github.com/rin-23/RobustSkinWeightsTransferCode
"""
import sys
import time

from maya import (
    cmds,
)

from maya.api import (
    OpenMaya as om,
    OpenMayaAnim as oma,
    # OpenMayaUI as omui,
)

import numpy as np
# import numpy.linalg as nplinalg
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.spatial import cKDTree

if sys.version_info > (3, 0):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import (
            Optional,  # noqa: F401
            Dict,  # noqa: F401
            List,  # noqa: F401
            Tuple,  # noqa: F401
            Pattern,  # noqa: F401
            Callable,  # noqa: F401
            Any,  # noqa: F401
            Text,  # noqa: F401
            Generator,  # noqa: F401
            Union  # noqa: F401
        )


from logging import (
    getLogger,
    WARN,  # noqa: F401
    DEBUG,  # noqa: F401
    INFO,  # noqa: F401
)
logger = getLogger(__name__)
logger.setLevel(INFO)


########################################################################################################################

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug("Execution time of {func.__name__}: {elapsed} seconds".format(
            func=func,
            elapsed=(end_time - start_time)
        ))
        return result
    return wrapper


########################################################################################################################
def as_selection_list(iterable):
    # type: (Union[Text, List[Text]]) -> om.MSelectionList
    """Converts an iterable to an MSelectionList."""

    selection_list = om.MSelectionList()
    if isinstance(iterable, (list, tuple)):
        for item in iterable:
            selection_list.add(item)
    else:
        selection_list.add(iterable)

    return selection_list


def as_dag_path(node):
    # type: (Union[Text, om.MObject, om.MDagPath]) -> om.MDagPath
    """Converts a node to an MDagPath."""

    if isinstance(node, om.MDagPath):
        return node

    if isinstance(node, om.MObject):
        dag_path = om.MDagPath.getAPathTo(node)
        return dag_path

    selection_list = as_selection_list(node)
    dag_path = selection_list.getDagPath(0)
    return dag_path


def as_depend_node(node):
    # type: (Union[Text, om.MObject, om.MDagPath]) -> om.MFnDependencyNode
    """Converts a node to an MFnDependencyNode."""

    selection_list = as_selection_list(node)
    depend_node = om.MFnDependencyNode(selection_list.getDependNode(0))

    return depend_node

def as_mfn_mesh(node):
    # type: (Union[Text, om.MObject, om.MDagPath]) -> om.MFnMesh
    """Converts a node to an MFnMesh."""

    dag_path = as_dag_path(node)
    mfn_mesh = om.MFnMesh(dag_path)
    return mfn_mesh


def as_mfn_skin_cluster(node):
    # type: (Union[Text, om.MObject, om.MDagPath]) -> om.MFnSkinCluster
    """Converts a node to an MFnSkinCluster."""

    dep_node = as_depend_node(node)
    mfn_skin_cluster = oma.MFnSkinCluster(dep_node.object())
    return mfn_skin_cluster


def load_meshes(source_mesh_name, target_mesh_name):
    # type: (str, str) -> Tuple[om.MFnMesh, om.MFnMesh]
    """load meshes."""

    print("Loading meshes...")

    if cmds.objectType(source_mesh_name) == "mesh":
        sm = source_mesh_name
    else:
        sm = cmds.listRelatives(source_mesh_name, shapes=True)[0]

    if cmds.objectType(target_mesh_name) == "mesh":
        tm = target_mesh_name
    else:
        tm = cmds.listRelatives(target_mesh_name, shapes=True)[0]

    source_mesh = as_mfn_mesh(sm)
    target_mesh = as_mfn_mesh(tm)

    return source_mesh, target_mesh
########################################################################################################################


@timeit
def get_vertex_positions_as_numpy_array(mesh):
    # type: (om.MFnMesh) -> np.ndarray
    """Returns numpy array of vertex positions."""

    points = mesh.getPoints(om.MSpace.kWorld)  # type: ignore
    return np.array([[p.x, p.y, p.z] for p in points])


@timeit
def get_vertex_normals_as_numpy_array(mesh):
    # type: (om.MFnMesh) -> np.ndarray
    """Returns numpy array of vertex normals."""

    normals = []
    for i in range(mesh.numVertices):
        normal = mesh.getVertexNormal(i, om.MSpace.kWorld)  # type: ignore
        normals.append([normal.x, normal.y, normal.z])
    return np.array(normals)


@timeit
def create_vertex_data_array(mesh):
    # type: (om.MFnMesh) -> np.ndarray
    """Create a structured numpy array containing vertex index, position, and normal."""

    vertex_data = np.zeros(
            mesh.numVertices,
            dtype=[
                ("index", np.int64),
                ("position", np.float64, 3),
                ("normal", np.float64, 3),
                ("face_index", np.int64),
            ])

    for i in range(mesh.numVertices):
        position = mesh.getPoint(i, om.MSpace.kWorld)  # type: ignore
        normal = mesh.getVertexNormal(i, om.MSpace.kWorld)  # type: ignore
        vertex_data[i] = (
                i,
                [position.x, position.y, position.z],
                [normal.x, normal.y, normal.z],
                -1)

    return vertex_data


@timeit
def get_closest_points(source_mesh, target_vertex_data):
    # type: (om.MFnMesh, np.ndarray) -> np.ndarray
    """get closest points and return a structured numpy array similar to target_vertex_data."""

    closest_points_data = np.zeros(target_vertex_data.shape, dtype=target_vertex_data.dtype)
    num_vertices = target_vertex_data.shape[0]

    if not cmds.about(batch=True):
        cmds.progressWindow(
                title="Finding closest points...",
                progress=0,
                status="Finding closest points...",
                isInterruptable=True,
                max=num_vertices,
        )

    for i in range(num_vertices):
        target_pos = target_vertex_data[i]["position"]
        
        # Get closest point on source mesh
        try:
            tmp = source_mesh.getClosestPointAndNormal(om.MPoint(target_pos), om.MSpace.kWorld)  # type: ignore
        except RuntimeError:
            continue

        closest_point, closest_normal, face_index = tmp
        pos = np.array([closest_point.x, closest_point.y, closest_point.z])
        norm = np.array([closest_normal.x, closest_normal.y, closest_normal.z])

        # Store target vertex index, closest point position, and closest point normal
        closest_points_data[i] = (
                target_vertex_data[i]["index"],
                pos,
                norm,
                face_index
                )

        if not cmds.about(batch=True):
            cmds.progressWindow(edit=True, step=1)

    if not cmds.about(batch=True):
        cmds.progressWindow(edit=True, endProgress=True)

    return closest_points_data


@timeit
def get_closest_points_by_kdtree(source_mesh, target_vertex_data):
    # type: (om.MFnMesh, np.ndarray) -> np.ndarray
    """get closest points and return a structured numpy array similar to target_vertex_data."""

    source_vertex_data = create_vertex_data_array(source_mesh)
    B_positions = np.array([vertex["position"] for vertex in source_vertex_data])
    A_positions = np.array([vertex["position"] for vertex in target_vertex_data])

    tree = cKDTree(B_positions)
    _, indices = tree.query(A_positions)

    nearest_in_B_for_A = source_vertex_data[indices]
    return nearest_in_B_for_A


@timeit
def filter_high_confidence_matches(target_vertex_data, closest_points_data, max_distance, max_angle):
    # type: (np.ndarray, np.ndarray, float, float) -> List[int]
    """filter high confidence matches using structured arrays."""

    target_positions = target_vertex_data["position"]
    target_normals = target_vertex_data["normal"]
    source_positions = closest_points_data["position"]
    source_normals = closest_points_data["normal"]

    # Calculate distances (vectorized)
    distances = np.linalg.norm(source_positions - target_positions, axis=1)

    # Calculate angles between normals (vectorized)
    cos_angles = np.einsum("ij,ij->i", source_normals, target_normals)
    cos_angles /= np.linalg.norm(source_normals, axis=1) * np.linalg.norm(target_normals, axis=1)
    cos_angles = np.abs(cos_angles)  # Consider opposite normals by taking absolute value
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi

    # Apply thresholds (vectorized)
    high_confidence_indices = np.where((distances <= max_distance) & (angles <= max_angle))[0]

    return high_confidence_indices.tolist()


@timeit
def copy_weights_for_confident_matches(source_mesh, target_mesh, confident_vertex_indices, closest_points_data):
    # type: (om.MFnMesh, om.MFnMesh, List[int], np.ndarray) -> Dict[int, np.ndarray]
    """copy weights for confident matches."""

    source_skin_cluster_name = get_skincluster(source_mesh.name())
    source_skin_cluster = as_mfn_skin_cluster(source_skin_cluster_name)
    deformer_bones = cmds.skinCluster(source_skin_cluster_name, query=True, influence=True)

    target_skin_cluster_name = get_or_create_skincluster(target_mesh.name(), deformer_bones)
    target_skin_cluster = as_mfn_skin_cluster(target_skin_cluster_name)

    known_weights = {}  # type: Dict[int, np.ndarray]

    # copy weights
    for i in confident_vertex_indices:
        src_face_index = closest_points_data[i]["face_index"]
        point = om.MPoint(closest_points_data[i]["position"])
        if src_face_index < 0:
            continue

        weights = get_weights_at_point(source_skin_cluster, source_mesh, src_face_index, point)

        if len(weights) <= 0:
            continue

        for j in range(len(weights)):
            cmds.setAttr(
                "{}.weightList[{}].weights[{}]".format(
                    target_skin_cluster.name(),
                    i,
                    j,
                ),
                weights[j])

        known_weights[i] = np.array(weights)

    return known_weights


@timeit
def transfer_weights(source_mesh, target_mesh, confident_vertex_indices=None):
    # type: (om.MFnMesh|Text, om.MFnMesh|Text, List[int]|None) -> None
    """transfer weights for confident matches."""

    if not isinstance(source_mesh, om.MFnMesh):
        source_mesh = as_mfn_mesh(source_mesh)

    if not isinstance(target_mesh, om.MFnMesh):
        target_mesh = as_mfn_mesh(target_mesh)

    src_skin_cluster_name = get_skincluster(source_mesh.name())
    src_deformer_bones = cmds.skinCluster(src_skin_cluster_name, query=True, influence=True)
    dst_skin_cluster_name = get_or_create_skincluster(target_mesh.name(), src_deformer_bones)
    dst_deformer_bones = cmds.skinCluster(src_skin_cluster_name, query=True, influence=True)

    if len(src_deformer_bones) != len(dst_deformer_bones):
        cmds.warning("The number of deformer bones is different between source and target meshes.")
        cmds.delete(dst_skin_cluster_name)
        dst_skin_cluster_name = get_or_create_skincluster(target_mesh.name(), src_deformer_bones)

    # TODO:
    # Should we only target confident_vertex_indices?
    cmds.copySkinWeights(
        sourceSkin=src_skin_cluster_name,  
        destinationSkin=dst_skin_cluster_name,
        noMirror=True,
        surfaceAssociation="closestPoint",
        influenceAssociation="closestJoint",
    )

@timeit
def get_skincluster(obj):
    # type: (Union[Text, List[Text]]) -> Text
    """get skincluster from object."""

    for history in cmds.listHistory(obj) or []:  # type: ignore
        obj_type = cmds.objectType(history)
        if obj_type == "skinCluster":
            return history

    raise RuntimeError("No skinCluster found on target mesh.")


@timeit
def get_or_create_skincluster(obj, deformers):
    # type: (Union[Text, List[Text]], List[Text]) -> Text
    try:
        return get_skincluster(obj)

    except RuntimeError:
        return cmds.skinCluster(
                obj,
                deformers,  # type: ignore
                toSelectedBones=True,
                tsb=True,
                mi=1,
                omi=True,
                bm=0,
                sm=0, nw=1, wd=0, rui=False, n=obj + "_skinCluster")[0]  # type: ignore


def get_weights_at_point(skin_cluster, mesh, face_index, closest_point):
    # type: (oma.MFnSkinCluster, om.MFnMesh, int, om.MPoint) -> np.ndarray
    """Returns weights at a given point."""

    it_face = om.MItMeshPolygon(mesh.dagPath())
    it_face.setIndex(int(face_index))
    num_vertices = it_face.polygonVertexCount()

    positions = it_face.getPoints(om.MSpace.kWorld)
    dastances = [positions[i].distanceTo(closest_point) for i in range(num_vertices)]
    total_distance = sum(dastances)
    ratios = [dastances[i] / total_distance for i in range(num_vertices)]
    reverse_ratios = [1.0 - ratios[i] for i in range(num_vertices)]
    reverse_total_normalized = sum(reverse_ratios)
    normalized_ratios = [reverse_ratios[i] / reverse_total_normalized for i in range(num_vertices)]

    n_vertices = it_face.polygonVertexCount()
    avg_weights = None

    for i in range(n_vertices):

        ratio = normalized_ratios[i]

        vertex_index = it_face.vertexIndex(i)
        component = om.MFnSingleIndexedComponent(om.MFnSingleIndexedComponent().create(om.MFn.kMeshVertComponent))
        component.addElement(int(vertex_index))
        raw_weights = skin_cluster.getWeights(mesh.dagPath(), component.object())[0]
        weights = np.array(raw_weights)

        if len(weights) <= 0:
            continue

        if avg_weights is None:
            avg_weights = weights * ratio

        else:
            avg_weights += (weights * ratio)

    return np.array(avg_weights)


def add_laplacian_entry_in_place(L, tri_positions, tri_indices):
    # type: (sp.lil_matrix, np.ndarray, np.ndarray) -> None
    """add laplacian entry.

    CAUTION: L is modified in-place.
    """

    i1 = tri_indices[0]
    i2 = tri_indices[1]
    i3 = tri_indices[2]

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]

    # calculate cotangent
    cotan1 = compute_cotangent(v2, v1, v3)
    cotan2 = compute_cotangent(v1, v2, v3)
    cotan3 = compute_cotangent(v1, v3, v2)

    # update laplacian matrix
    L[i1, i2] += cotan1  # type: ignore
    L[i2, i1] += cotan1  # type: ignore
    L[i1, i1] -= cotan1  # type: ignore
    L[i2, i2] -= cotan1  # type: ignore

    L[i2, i3] += cotan2  # type: ignore
    L[i3, i2] += cotan2  # type: ignore
    L[i2, i2] -= cotan2  # type: ignore
    L[i3, i3] -= cotan2  # type: ignore

    L[i1, i3] += cotan3  # type: ignore
    L[i3, i1] += cotan3  # type: ignore
    L[i1, i1] -= cotan3  # type: ignore
    L[i3, i3] -= cotan3  # type: ignore


def add_area_in_place(areas, tri_positions, tri_indices):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> None
    """add area.

    CAUTION: areas is modified in-place.
    """

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    for idx in tri_indices:
        areas[idx] += area


def compute_laplacian_and_mass_matrix(mesh):
    # type: (om.MFnMesh) -> Tuple[sp.csr_array, sp.dia_matrix]
    """compute laplacian matrix from mesh.

    treat area as mass matrix.
    """

    # initialize sparse laplacian matrix
    n_vertices = mesh.numVertices
    L = sp.lil_matrix((n_vertices, n_vertices))
    areas = np.zeros(n_vertices)

    # for each edge and face, calculate the laplacian entry and area
    face_iter = om.MItMeshPolygon(mesh.dagPath())
    while not face_iter.isDone():

        n_tri = face_iter.numTriangles()

        for j in range(n_tri):

            tri_positions, tri_indices = face_iter.getTriangle(j)
            add_laplacian_entry_in_place(L, tri_positions, tri_indices)
            add_area_in_place(areas, tri_positions, tri_indices)

        face_iter.next()

    L_csr = L.tocsr()
    M_csr = sp.diags(areas)

    return L_csr, M_csr


def compute_cotangent(v1, v2, v3):
    # type: (om.MPoint, om.MPoint, om.MPoint) -> float
    """compute cotangent from three points."""

    edeg1 = v2 - v1
    edeg2 = v3 - v1

    norm1 = edeg1 ^ edeg2

    area = norm1.length()
    cotan = edeg1 * edeg2 / area

    return cotan


def compute_mass_matrix(mesh):
    # type: (om.MFnMesh) -> sp.dia_matrix
    """Compute the mass matrix for a given mesh.

    This function calculates the mass matrix of a mesh by iterating over its faces. 
    For each face, it computes the area of the triangle formed by the vertices of the face.
    The area is then assigned to the corresponding vertices.

    The mass matrix is represented as a diagonal sparse matrix, where each
    diagonal element corresponds to the sum of the areas of all faces
    connected to a vertex.

    Parameters:
        mesh (om.MFnMesh): The mesh for which the mass matrix is to be computed.

    Returns:
        sp.dia_matrix: The diagonal sparse mass matrix, where each diagonal element represents 
                       the total area associated with a vertex.
    """

    n_vertices = mesh.numVertices
    areas = np.zeros(n_vertices)
    face_iter = om.MItMeshPolygon(mesh.dagPath())

    while not face_iter.isDone():

        tri_positions, tri_indices = face_iter.getTriangle(0)
        v1 = np.array(tri_positions[0])
        v2 = np.array(tri_positions[1])
        v3 = np.array(tri_positions[2])

        # calculate area of the current face
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

        # add area to the corresponding vertices
        for idx in tri_indices:
            areas[idx] += area

        face_iter.next()

    # create sparse diagonal mass matrix
    M = sp.diags(areas)

    return M


def __do_inpainting(mesh, known_weights):
    # type: (om.MFnMesh, Dict[int, np.ndarray]) -> np.ndarray

    L, M = compute_laplacian_and_mass_matrix(mesh)
    # Q = -L + L @ sp.diags(np.reciprocal(M.diagonal())) @ L  # @ operator is not supported for python 2...
    Q = -L + np.dot(L, sp.diags(np.reciprocal(M.diagonal())).dot(L))

    S_match = np.array(list(known_weights.keys()))
    S_nomatch = np.array(list(set(range(mesh.numVertices)) - set(S_match)))

    Q_UU = sp.csr_matrix(Q[np.ix_(S_nomatch, S_nomatch)])
    Q_UI = sp.csr_matrix(Q[np.ix_(S_nomatch, S_match)])

    num_vertices = mesh.numVertices
    num_bones = len(next(iter(known_weights.values())))

    W = np.zeros((num_vertices, num_bones))
    for i, weights in known_weights.items():
        W[i] = weights

    W_I = W[S_match, :]
    W_U = W[S_nomatch, :]

    for bone_idx in range(num_bones):
        # b = -Q_UI @ W_I[:, bone_idx]  # @ operator is not supported for python 2...
        b = -np.dot(Q_UI, W_I[:, bone_idx])
        W_U[:, bone_idx] = splinalg.spsolve(Q_UU, b)

    W[S_nomatch, :] = W_U

    # apply constraints,

    # each element is between 0 and 1
    W = np.clip(W, 0.0, 1.0)

    # normalize each row to sum to 1
    W = W / W.sum(axis=1, keepdims=True)

    return W


def calculate_inpainting(mesh, unknown_vertex_indices):
    # type: (om.MFnMesh, List[int]|Tuple[int]) -> np.ndarray
    """Inpainting weights for unknown vertices from known vertices."""

    num_vertices = mesh.numVertices
    known_indices = list(set(range(num_vertices)) - set(unknown_vertex_indices))
    skin_cluster_name = get_skincluster(mesh.name())
    skin_cluster = as_mfn_skin_cluster(skin_cluster_name)

    weights, num_deformers = skin_cluster.getWeights(mesh.dagPath(), om.MObject())
    weights_np = np.array(weights)

    known_weights = {}  # type: Dict[int, np.ndarray]
    for vertex_index in known_indices:
        known_weights[vertex_index] = weights_np[vertex_index * num_deformers: (vertex_index + 1) * num_deformers]

    return __do_inpainting(mesh, known_weights)


def compute_weights_for_remaining_vertices(target_mesh, known_weights):
    # type: (om.MFnMesh, Dict[int, np.ndarray]) -> np.ndarray
    """compute weights for remaining vertices."""
   
    try:
        optimized = __do_inpainting(target_mesh, known_weights)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error: {}".format(e))
        raise

    return optimized


def apply_weight_inpainting(target_mesh, optimized_weights, unconvinced_vertex_indices):
    # type: (om.MFnMesh, np.ndarray, List[int]) -> None
    """apply weight inpainting."""

    target_skin_cluster_name = get_skincluster(target_mesh.name())
    target_skin_cluster = as_mfn_skin_cluster(target_skin_cluster_name)

    for i in unconvinced_vertex_indices:

        weights = optimized_weights[i]

        if len(weights) <= 0:
            continue

        for j in range(len(weights)):
            cmds.setAttr(
                "{}.weightList[{}].weights[{}]".format(
                    target_skin_cluster.name(),
                    i,
                    j,
                ),
                weights[j])

    print("Done.")


def calculate_threshold_distance(mesh, threadhold_ratio=0.05):
    # type: (om.MFnMesh, float) -> float
    """Returns dbox * 0.05

    dbox is the target mesh bounding box diagonal length.
    """

    bbox = mesh.boundingBox
    bbox_min = bbox.min
    bbox_max = bbox.max
    bbox_diag = bbox_max - bbox_min
    bbox_diag_length = bbox_diag.length()

    threshold_distance = bbox_diag_length * threadhold_ratio

    return threshold_distance


def segregate_vertices_by_confidence(src_mesh, dst_mesh, threshold_distance=0.05, threshold_angle=25.0, use_kdtree=False):
    # type: (om.MFnMesh|Text, om.MFnMesh|Text, float, float, bool) -> Tuple[List[int], List[int]]
    """segregate vertices by confidence."""

    if not isinstance(src_mesh, om.MFnMesh):
        src_mesh = as_mfn_mesh(src_mesh)

    if not isinstance(dst_mesh, om.MFnMesh):
        dst_mesh = as_mfn_mesh(dst_mesh)

    threshold_distance = calculate_threshold_distance(dst_mesh, threshold_distance)
    target_vertex_data = create_vertex_data_array(dst_mesh)

    if use_kdtree:
        closest_points_data = get_closest_points_by_kdtree(src_mesh, target_vertex_data)
    else:
        closest_points_data = get_closest_points(src_mesh, target_vertex_data)

    confident_vertex_indices = filter_high_confidence_matches(target_vertex_data, closest_points_data, threshold_distance, threshold_angle)
    unconvinced_vertex_indices = list(set(range(dst_mesh.numVertices)) - set(confident_vertex_indices))

    return confident_vertex_indices, unconvinced_vertex_indices


def inpaint_weights(target_mesh, indices):
    # type: (om.MFnMesh|Text, List[int]) -> None
    """apply inpainting for indices."""

    if not isinstance(target_mesh, om.MFnMesh):
        target_mesh = as_mfn_mesh(target_mesh)

    tmp = calculate_inpainting(target_mesh, indices)
    apply_weight_inpainting(target_mesh, tmp, indices)


def main():

    # setup
    source_mesh, target_mesh = load_meshes(cmds.ls(sl=True)[0], cmds.ls(sl=True)[1])
    tmp = segregate_vertices_by_confidence(source_mesh, target_mesh)
    target_vertex_data = create_vertex_data_array(target_mesh)

    # confidence
    confident_vertex_indices = tmp[0]
    unconvinced_vertex_indices = tmp[1]

    closest_points_data = get_closest_points(source_mesh, target_vertex_data)
    known_weights = copy_weights_for_confident_matches(source_mesh, target_mesh, confident_vertex_indices, closest_points_data)

    # inpainting
    optimized_weights = compute_weights_for_remaining_vertices(target_mesh, known_weights)
    apply_weight_inpainting(target_mesh, optimized_weights, unconvinced_vertex_indices)


if __name__ == "__main__":
    main()
