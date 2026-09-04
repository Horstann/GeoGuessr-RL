import aiohttp
import ssl
import certifi
import json
from pathlib import Path
from tqdm import tqdm
import shutil
import yaml
import tempfile
import asyncio
import math
import random
from dataclasses import dataclass
from streetlevel import streetview
from huggingface_hub import CommitOperationAdd, HfApi
import cv2
import numpy as np
from PIL import Image

with open("./conf/ai_keys/default.yaml", "r") as f:
    config = yaml.safe_load(f)
hf_token = config["huggingface_token"]
api = HfApi(token=hf_token)

HF_REPO_ID = config["huggingface_dataset_id"]
HF_REPO_TYPE = "dataset"
ZOOM = 5
JPEG_QUALITY = 80

LOOKUP_CONCURRENCY = 10
LOOKUP_MAX_ATTEMPTS = 4
LOOKUP_RETRY_BASE_DELAY = 1.0
LOOKUP_RETRY_JITTER = 0.5
PANORAMA_CONCURRENCY = 7
HTTP_CONNECTION_LIMIT = 36
GRAPH_BATCH_SIZE = 8

SOURCE_GRAPH_DIR = Path("data/json_graphs_old_")
REFRESHED_GRAPH_DIR = Path("data/json_graphs")
PROCESSED_GRAPH_DIR = Path("data/json_graphs_old")
REMOTE_IMAGE_DIR = "pano_images_02"
REMOTE_GRAPH_DIR = "json_graphs"

REFRESHED_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_GRAPH_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PreparedBatch:
    """Files that must remain on disk until their Hugging Face commit ends."""

    temporary_directories: list
    operations: list
    remote_graph_paths: set
    source_graphs: list

    def cleanup(self):
        for temporary_directory in self.temporary_directories:
            temporary_directory.cleanup()


# =====================================================================
# Helpers
# =====================================================================

def crop_black_edges(image: Image.Image, threshold=10) -> Image.Image:
    """
    Automatically crop black edges from an image.
    
    Args:
        image (Image.Image): Input PIL Image
        threshold (int): Threshold for what is considered "black" (0-255)
    
    Returns:
        Image.Image: Cropped PIL Image
    """
    
    # Convert PIL Image to cv2 format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not convert image")
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask where non-black pixels are True
    mask = gray > threshold
    
    # Find coordinates of non-black pixels
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        print("Warning: Image appears to be entirely black")
        return img
    
    # Get bounding box of non-black content
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop the image
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    
    # Convert back to PIL Image
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)
    
    return cropped_pil


def calculate_bearing_radians(source_coordinate, target_coordinate):
    """Return the initial compass bearing, or None for coincident points."""
    lat1 = math.radians(source_coordinate["lat"])
    lon1 = math.radians(source_coordinate["lon"])
    lat2 = math.radians(target_coordinate["lat"])
    lon2 = math.radians(target_coordinate["lon"])

    if math.isclose(lat1, lat2, abs_tol=1e-12) and math.isclose(
        lon1, lon2, abs_tol=1e-12
    ):
        return None

    delta_lon = lon2 - lon1
    y = math.sin(delta_lon) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    return math.atan2(y, x) % (2 * math.pi)


def find_connected_components(adjacency_matrix):
    """Return weakly connected components for a directed adjacency matrix."""
    unvisited = set(range(len(adjacency_matrix)))
    components = []

    while unvisited:
        start = unvisited.pop()
        component = {start}
        stack = [start]

        while stack:
            source_id = stack.pop()
            neighbors = {
                target_id
                for target_id in range(len(adjacency_matrix))
                if target_id != source_id
                and (
                    adjacency_matrix[source_id][target_id] != -1
                    or adjacency_matrix[target_id][source_id] != -1
                )
            }
            new_neighbors = neighbors & unvisited
            unvisited -= new_neighbors
            component |= new_neighbors
            stack.extend(new_neighbors)

        components.append(sorted(component))

    return components


def remove_unresolved_nodes(graph, unresolved_matrix_ids):
    """Remove unresolved nodes, resize the matrix, and verify connectivity."""
    nodes = graph["nodes"]
    adjacency_matrix = graph["adjacency_matrix"]
    node_count = len(nodes)

    if len(adjacency_matrix) != node_count or any(
        len(row) != node_count for row in adjacency_matrix
    ):
        raise ValueError("adjacency_matrix must be square and match the node count")

    matrix_ids = [node["matrix_id"] for node in nodes]
    if sorted(matrix_ids) != list(range(node_count)):
        raise ValueError("matrix_id values must uniquely cover 0 through node_count - 1")

    unresolved_matrix_ids = set(unresolved_matrix_ids)
    unknown_matrix_ids = unresolved_matrix_ids - set(matrix_ids)
    if unknown_matrix_ids:
        raise ValueError(f"Unknown unresolved matrix IDs: {sorted(unknown_matrix_ids)}")

    if not unresolved_matrix_ids:
        return {
            "removed_pano_ids": [],
            "old_to_new_matrix_id": {matrix_id: matrix_id for matrix_id in matrix_ids},
            "empty_graph": False,
        }

    removed_nodes = [
        node for node in nodes if node["matrix_id"] in unresolved_matrix_ids
    ]
    removed_pano_ids = [node["pano_id"] for node in removed_nodes]
    kept_nodes = [
        node for node in nodes if node["matrix_id"] not in unresolved_matrix_ids
    ]
    if not kept_nodes:
        return {
            "removed_pano_ids": removed_pano_ids,
            "old_to_new_matrix_id": {},
            "empty_graph": True,
        }

    if not any(
        node["pano_id"] == graph["center_pano_id"] for node in kept_nodes
    ):
        raise RuntimeError(f"The center panorama {graph['center_pano_id']} could not be resolved")

    kept_old_matrix_ids = [node["matrix_id"] for node in kept_nodes]
    graph["adjacency_matrix"] = [
        [adjacency_matrix[source_id][target_id] for target_id in kept_old_matrix_ids]
        for source_id in kept_old_matrix_ids
    ]

    old_to_new_matrix_id = {}
    for new_matrix_id, node in enumerate(kept_nodes):
        old_to_new_matrix_id[node["matrix_id"]] = new_matrix_id
        node["matrix_id"] = new_matrix_id
    graph["nodes"] = kept_nodes

    components = find_connected_components(graph["adjacency_matrix"])
    if len(components) > 1:
        graph["component_sizes"] = [
            len(component) for component in components
        ]
    else:
        graph.pop("component_sizes", None)

    return {
        "removed_pano_ids": removed_pano_ids,
        "old_to_new_matrix_id": old_to_new_matrix_id,
        "empty_graph": False,
    }


def contract_graph_and_update_directions(graph, replaced_matrix_ids):
    """Merge duplicate pano IDs and selectively refresh edge directions."""
    nodes = graph["nodes"]
    adjacency_matrix = graph["adjacency_matrix"]
    node_count = len(nodes)

    if len(adjacency_matrix) != node_count or any(
        len(row) != node_count for row in adjacency_matrix
    ):
        raise ValueError("adjacency_matrix must be square and match the node count")

    matrix_ids = [node["matrix_id"] for node in nodes]
    if sorted(matrix_ids) != list(range(node_count)):
        raise ValueError("matrix_id values must uniquely cover 0 through node_count - 1")

    groups = []
    group_by_pano_id = {}
    merged_pano_ids = []

    for node in nodes:
        pano_id = node["pano_id"]
        if pano_id in group_by_pano_id:
            group_by_pano_id[pano_id]["old_matrix_ids"].append(node["matrix_id"])
            merged_pano_ids.append(pano_id)
            continue

        group = {
            "node": node,
            "representative_matrix_id": node["matrix_id"],
            "old_matrix_ids": [node["matrix_id"]],
        }
        group_by_pano_id[pano_id] = group
        groups.append(group)

    merged_matrix = [[-1 for _ in groups] for _ in groups]
    recomputed_edge_count = 0
    coincident_edges = []
    for source_index, source_group in enumerate(groups):
        for target_index, target_group in enumerate(groups):
            if source_index == target_index:
                continue

            selected_edge = None
            # Prefer the representative-to-representative edge, then fall
            # back to an edge inherited from either duplicate group.
            for source_id in source_group["old_matrix_ids"]:
                for target_id in target_group["old_matrix_ids"]:
                    direction = adjacency_matrix[source_id][target_id]
                    if direction != -1:
                        selected_edge = (source_id, target_id, direction)
                        break
                if selected_edge is not None:
                    break

            if selected_edge is None:
                continue

            source_id, target_id, direction = selected_edge
            inherited_edge = (
                source_id != source_group["representative_matrix_id"]
                or target_id != target_group["representative_matrix_id"]
            )
            endpoint_replaced = (
                source_id in replaced_matrix_ids or target_id in replaced_matrix_ids
            )

            if inherited_edge or endpoint_replaced:
                updated_direction = calculate_bearing_radians(
                    source_group["node"]["coordinate"],
                    target_group["node"]["coordinate"],
                )
                if updated_direction is None:
                    coincident_edges.append(
                        (source_group["node"]["pano_id"], target_group["node"]["pano_id"])
                    )
                else:
                    direction = updated_direction
                    recomputed_edge_count += 1

            merged_matrix[source_index][target_index] = direction

    kept_nodes = [group["node"] for group in groups]
    for new_matrix_id, node in enumerate(kept_nodes):
        node["matrix_id"] = new_matrix_id
    graph["nodes"] = kept_nodes
    graph["adjacency_matrix"] = merged_matrix
    return {
        "merged_pano_ids": merged_pano_ids,
        "recomputed_edge_count": recomputed_edge_count,
        "coincident_edges": coincident_edges,
    }


# =====================================================================
# Progress-aware asynchronous gather
# =====================================================================

async def gather_with_progress(coroutines, description, progress_after=None):
    """
    Run all supplied coroutines concurrently and display a progress bar.

    All tasks are awaited before an exception is raised, preventing temporary
    directories from being deleted while another task is still writing.
    """
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    if not tasks: return []
    results = []
    errors = []
    progress_bar = None

    if progress_after is None or progress_after.done():
        progress_bar = tqdm(total=len(tasks), desc=description)

    try:
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                results.append(result)
            except Exception as error:
                errors.append(error)

            if progress_bar is None and progress_after.done():
                progress_bar = tqdm(
                    total=len(tasks),
                    initial=len(results) + len(errors),
                    desc=description,
                )
            elif progress_bar is not None:
                progress_bar.update(1)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if errors:
        error_messages = "\n".join(str(error) for error in errors)
        raise RuntimeError(f"{len(errors)} asynchronous task(s) failed.\n{error_messages}")
    return results


# =====================================================================
# Metadata lookup
# =====================================================================

async def lookup_with_retries(operation, description, lookup_semaphore):
    """Run one metadata lookup with exponential backoff on transient errors."""
    transient_errors = (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        json.JSONDecodeError,
    )

    for attempt in range(1, LOOKUP_MAX_ATTEMPTS + 1):
        try:
            async with lookup_semaphore:
                return await operation()
        except transient_errors as error:
            if attempt == LOOKUP_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{description} failed after {LOOKUP_MAX_ATTEMPTS} "
                    f"attempts: {error}"
                ) from error

            delay = (
                LOOKUP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                + random.uniform(0, LOOKUP_RETRY_JITTER)
            )
            tqdm.write(
                f"Retrying {description} after attempt {attempt}/"
                f"{LOOKUP_MAX_ATTEMPTS} failed ({error}); waiting "
                f"{delay:.1f}s."
            )
            # Sleep after releasing the semaphore so another lookup can run.
            await asyncio.sleep(delay)


async def resolve_node(index, node, session, lookup_semaphore):
    """
    Resolve the panorama metadata for one graph node.

    First tries the existing pano ID. If that ID is stale, searches using
    the node's coordinates.
    """
    old_pano_id = node["pano_id"]
    old_lat = node["coordinate"]["lat"]
    old_lon = node["coordinate"]["lon"]

    pano = await lookup_with_retries(
        operation=lambda: streetview.find_panorama_by_id_async(
            panoid=old_pano_id,
            session=session,
        ),
        description=f"panorama ID lookup for {old_pano_id}",
        lookup_semaphore=lookup_semaphore,
    )
    used_coordinate_fallback = pano is None
    if used_coordinate_fallback:
        pano = await lookup_with_retries(
            operation=lambda: streetview.find_panorama_async(
                lat=old_lat,
                lon=old_lon,
                session=session,
            ),
            description=(
                f"coordinate lookup for {old_pano_id} "
                f"at ({old_lat}, {old_lon})"
            ),
            lookup_semaphore=lookup_semaphore,
        )

    return {
        "index": index,
        "pano": pano,
        "old_pano_id": old_pano_id,
        "old_lat": old_lat,
        "old_lon": old_lon,
        "used_coordinate_fallback": used_coordinate_fallback,
    }


# =====================================================================
# Panorama downloading
# =====================================================================

async def download_pano(pano, remote_image_path, pano_dir, session, panorama_semaphore):
    """
    Download, stitch, compress, and save one panorama.

    The semaphore limits the number of large panorama images held in RAM
    simultaneously.
    """
    local_image_path = pano_dir / f"{pano.id}.jpg"
    image = None
    cropped_image = None

    async with panorama_semaphore:
        try:
            image = await streetview.get_panorama_async(pano=pano, session=session, zoom=ZOOM)
            cropped_image = await asyncio.to_thread(crop_black_edges, image)

            # Pillow's JPEG compression is synchronous, so move it to a worker thread to avoid blocking the asyncio event loop.
            await asyncio.to_thread(
                cropped_image.save,
                local_image_path,
                "JPEG",
                quality=JPEG_QUALITY,
                optimize=True,
            )
        finally:
            if cropped_image is not None and cropped_image is not image:
                cropped_image.close()
            if image is not None:
                image.close()

    return remote_image_path, local_image_path


# =====================================================================
# Graph processing
# =====================================================================

async def prepare_graph(graph_file, session, known_files, progress_after=None):
    print(f"\nProcessing graph: {graph_file.name}")

    with graph_file.open("r", encoding="utf-8") as file:
        graph = json.load(file)

    temporary_directory = tempfile.TemporaryDirectory(prefix="streetview_batch_")
    try:
        temp_dir = temporary_directory.name
        batch_dir = Path(temp_dir)
        batch_graph_dir = batch_dir / "json_graphs"
        pano_dir = batch_dir / "pano_images"
        batch_graph_dir.mkdir(parents=True)
        pano_dir.mkdir(parents=True)

        # -------------------------------------------------------------
        # Stage 1: Resolve all node metadata concurrently
        # -------------------------------------------------------------

        lookup_semaphore = asyncio.Semaphore(LOOKUP_CONCURRENCY)
        lookup_coroutines = [
            resolve_node(index=i, node=node, session=session, lookup_semaphore=lookup_semaphore)
            for i, node in enumerate(graph["nodes"])
        ]
        resolved_nodes = await gather_with_progress(
            lookup_coroutines,
            description=f"Looking up {graph_file.name}",
            progress_after=progress_after,
        )
        # as_completed returns results in completion order.
        resolved_nodes.sort(key=lambda result: result["index"])
        unresolved_results = [result for result in resolved_nodes if result["pano"] is None]
        unresolved_matrix_ids = {
            graph["nodes"][result["index"]]["matrix_id"]
            for result in unresolved_results
        }

        # -------------------------------------------------------------
        # Stage 2: Update JSON and collect unique new panoramas
        # -------------------------------------------------------------

        panos_to_download = {}
        replaced_matrix_ids = set()

        for result in resolved_nodes:
            i = result["index"]
            pano = result["pano"]
            old_pano_id = result["old_pano_id"]
            old_lat = result["old_lat"]
            old_lon = result["old_lon"]

            if pano is None:
                continue

            node = graph["nodes"][i]
            matrix_id = node["matrix_id"]
            pano_id_changed = old_pano_id != pano.id

            if pano_id_changed:
                replaced_matrix_ids.add(matrix_id)
                if old_pano_id == graph["center_pano_id"]:
                    graph["center_pano_id_old"] = old_pano_id
                    graph["center_pano_id"] = pano.id

            # Normalize every node to the authoritative metadata returned
            # by the current Street View lookup.
            coordinate = {
                "lat": pano.lat,
                "lon": pano.lon,
                "heading": pano.heading,
                "roll": pano.roll,
                "pitch": pano.pitch,
            }
            if pano_id_changed:
                coordinate.update({"lat_old": old_lat, "lon_old": old_lon})
                node.update({"pano_id_old": old_pano_id})
            node.update({"pano_id": pano.id, "coordinate": coordinate})
            # Convert LocalizedString objects into string
            node["address"] = ", ".join(
                part.value for part in (pano.address or [])
            )

            remote_image_path = f"{REMOTE_IMAGE_DIR}/{pano.id}.jpg"

            # Do not download images already present on Hugging Face.
            if remote_image_path in known_files:
                continue

            # Prevent duplicate pano IDs within the same graph.
            panos_to_download.setdefault(remote_image_path, pano)

        removal_update = remove_unresolved_nodes(graph, unresolved_matrix_ids)
        if removal_update["empty_graph"]:
            print(f"Skipping {graph_file.name}: no panoramas could be resolved.")
            temporary_directory.cleanup()
            return None
        if removal_update["removed_pano_ids"]:
            print(
                f"Removed {len(removal_update['removed_pano_ids'])} unresolved "
                f"node(s): {', '.join(removal_update['removed_pano_ids'])}"
            )
        replaced_matrix_ids = {
            removal_update["old_to_new_matrix_id"][matrix_id]
            for matrix_id in replaced_matrix_ids
        }

        graph_update = contract_graph_and_update_directions(graph, replaced_matrix_ids)
        if graph_update["merged_pano_ids"]:
            print(
                f"Merged {len(graph_update['merged_pano_ids'])} duplicate node "
                f"occurrence(s): {', '.join(graph_update['merged_pano_ids'])}"
            )
        if graph_update["recomputed_edge_count"]:
            print(
                f"Recomputed {graph_update['recomputed_edge_count']} "
                "directed edge angle(s)."
            )
        for source_pano_id, target_pano_id in graph_update["coincident_edges"]:
            print(
                f"Warning: retained the original angle for coincident "
                f"panoramas {source_pano_id} -> {target_pano_id}."
            )

        refreshed_graph_name = f"{graph['center_pano_id']}_10_graph.json"
        refreshed_graph_path = REFRESHED_GRAPH_DIR / refreshed_graph_name
        batch_graph_path = batch_graph_dir / refreshed_graph_name
        remote_graph_path = f"{REMOTE_GRAPH_DIR}/{refreshed_graph_name}"

        if remote_graph_path in known_files:
            raise FileExistsError(
                f"Refusing to overwrite existing or staged graph: {remote_graph_path}"
            )

        # -------------------------------------------------------------
        # Stage 3: Download panoramas concurrently
        # -------------------------------------------------------------

        panorama_semaphore = asyncio.Semaphore(PANORAMA_CONCURRENCY)
        download_coroutines = [
            download_pano(pano=pano, remote_image_path=remote_image_path, pano_dir=pano_dir, session=session, panorama_semaphore=panorama_semaphore)
            for remote_image_path, pano in panos_to_download.items()
        ]
        downloaded_images = await gather_with_progress(
            download_coroutines,
            description=f"Downloading {graph_file.name}",
            progress_after=progress_after,
        )
        pending_images = dict(downloaded_images)

        # -------------------------------------------------------------
        # Stage 4: Save updated graph locally
        # -------------------------------------------------------------

        with refreshed_graph_path.open("w", encoding="utf-8") as output:
            json.dump(graph, output, indent=2, ensure_ascii=False)
        shutil.copy2(refreshed_graph_path, batch_graph_path)

        # -------------------------------------------------------------
        # Stage 5: Prepare append-only Hugging Face operations
        # -------------------------------------------------------------

        operations = [
            CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=local_path,
            )
            for remote_path, local_path in pending_images.items()
        ]

        operations.append(
            CommitOperationAdd(
                path_in_repo=remote_graph_path,
                path_or_fileobj=batch_graph_path,
            )
        )

        # Reserve these paths immediately. This prevents later graphs, including
        # graphs prepared while this batch uploads, from downloading duplicates.
        known_files.update(operation.path_in_repo for operation in operations)

        return {
            "temporary_directory": temporary_directory,
            "operations": operations,
            "remote_graph_path": remote_graph_path,
            "source_graph": graph_file,
        }
    except BaseException:
        temporary_directory.cleanup()
        raise


# =====================================================================
# Batch preparation and upload
# =====================================================================

async def prepare_batch(
    graph_files, session, known_files, progress_after=None
):
    """Prepare several graphs while keeping all staged files alive."""
    prepared_graphs = []

    for graph_file in graph_files:
        processed_path = PROCESSED_GRAPH_DIR / graph_file.name
        if processed_path.exists():
            for completed_graph in prepared_graphs:
                completed_graph["temporary_directory"].cleanup()
            raise FileExistsError(
                f"The source graph cannot be archived because this file "
                f"already exists: {processed_path}"
            )

        try:
            prepared_graph = await prepare_graph(
                graph_file=graph_file,
                session=session,
                known_files=known_files,
                progress_after=progress_after,
            )
        except BaseException as error:
            print(f"Failed to process {graph_file.name}: {error}")
            for completed_graph in prepared_graphs:
                completed_graph["temporary_directory"].cleanup()
            raise

        if prepared_graph is not None:
            prepared_graphs.append(prepared_graph)

    if not prepared_graphs:
        return None

    return PreparedBatch(
        temporary_directories=[
            graph["temporary_directory"] for graph in prepared_graphs
        ],
        operations=[
            operation
            for graph in prepared_graphs
            for operation in graph["operations"]
        ],
        remote_graph_paths={
            graph["remote_graph_path"] for graph in prepared_graphs
        },
        source_graphs=[graph["source_graph"] for graph in prepared_graphs],
    )


def commit_batch(batch):
    """Commit one prepared batch. This function runs in a worker thread."""
    repo_info = api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    parent_commit = repo_info.sha
    latest_remote_files = set(
        api.list_repo_files(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            revision=parent_commit,
        )
    )

    collisions = {
        operation.path_in_repo
        for operation in batch.operations
        if operation.path_in_repo in latest_remote_files
    }
    graph_collisions = batch.remote_graph_paths & collisions
    if graph_collisions:
        collision_list = ", ".join(sorted(graph_collisions))
        raise FileExistsError(
            f"The following graph(s) appeared on Hugging Face while this "
            f"batch was processing: {collision_list}. The commit was "
            "cancelled to prevent an overwrite."
        )

    # Images uploaded by another process can safely be reused. Graph JSONs
    # remain append-only and were rejected above if they collided.
    operations = [
        operation
        for operation in batch.operations
        if operation.path_in_repo not in collisions
    ]
    number_of_new_images = sum(
        operation.path_in_repo.startswith(f"{REMOTE_IMAGE_DIR}/")
        for operation in operations
    )
    graph_count = len(batch.remote_graph_paths)

    commit = api.create_commit(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        operations=operations,
        parent_commit=parent_commit,
        commit_message=f"Add {graph_count} refreshed graph(s) with {number_of_new_images} new panoramas",
    )
    return {
        "commit_url": commit.commit_url,
        "latest_remote_files": latest_remote_files,
        "committed_paths": {
            operation.path_in_repo for operation in operations
        },
    }


async def finish_upload(batch, upload_task, known_files):
    """Wait for a commit, archive its sources, and release staged files."""
    cancellation_error = None
    try:
        try:
            result = await asyncio.shield(upload_task)
        except asyncio.CancelledError as error:
            # Cancelling to_thread only cancels the awaitable; its worker thread
            # keeps running. Wait for it before deleting the staged files.
            cancellation_error = error
            result = await upload_task

        processed_paths = [
            PROCESSED_GRAPH_DIR / source_graph.name
            for source_graph in batch.source_graphs
        ]
        existing_paths = [path for path in processed_paths if path.exists()]
        if existing_paths:
            path_list = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(
                "The upload succeeded, but its source graphs could not be "
                f"archived because these files already exist: {path_list}"
            )

        for source_graph, processed_path in zip(
            batch.source_graphs, processed_paths
        ):
            source_graph.replace(processed_path)

        known_files.update(result["latest_remote_files"])
        known_files.update(result["committed_paths"])
        print(
            f"Uploaded and archived {len(batch.source_graphs)} graph(s): "
            f"{result['commit_url']}"
        )
        if cancellation_error is not None:
            raise cancellation_error
    finally:
        batch.cleanup()


# =====================================================================
# Main
# =====================================================================

async def main():
    source_graphs = sorted(SOURCE_GRAPH_DIR.glob("*_graph.json"))
    if not source_graphs:
        print(f"No JSON files found in {SOURCE_GRAPH_DIR.resolve()}")
        return
    print(f"Found {len(source_graphs)} graph(s).")

    known_files = set(api.list_repo_files(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
    ))
    print(f"Found {len(known_files)} existing remote file(s).")

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(
        ssl=ssl_context,
        limit=HTTP_CONNECTION_LIMIT,
        limit_per_host=HTTP_CONNECTION_LIMIT,
        ttl_dns_cache=300,
    )

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        uploading_batch = None
        upload_task = None

        for batch_start in range(0, len(source_graphs), GRAPH_BATCH_SIZE):
            graph_files = source_graphs[
                batch_start:batch_start + GRAPH_BATCH_SIZE
            ]

            try:
                prepared_batch = await prepare_batch(
                    graph_files=graph_files,
                    session=session,
                    known_files=known_files,
                    progress_after=upload_task,
                )
            except BaseException:
                # A previous commit may have completed while preparation failed.
                # Always observe and finalize it before leaving the program.
                if upload_task is not None:
                    await finish_upload(
                        uploading_batch, upload_task, known_files
                    )
                raise

            # Preparation above overlaps with the preceding batch's upload.
            # Before starting another commit, finish and archive that batch.
            if upload_task is not None:
                try:
                    await finish_upload(
                        uploading_batch, upload_task, known_files
                    )
                except BaseException:
                    if prepared_batch is not None:
                        prepared_batch.cleanup()
                    raise

            uploading_batch = prepared_batch
            upload_task = None
            if uploading_batch is not None:
                print(
                    f"Uploading {len(uploading_batch.source_graphs)} graph(s) "
                    "in the background."
                )
                upload_task = asyncio.create_task(
                    asyncio.to_thread(commit_batch, uploading_batch)
                )

        if upload_task is not None:
            await finish_upload(uploading_batch, upload_task, known_files)


if __name__ == "__main__":
    asyncio.run(main())
