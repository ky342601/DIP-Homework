import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """
    if len(source_pts) < 3 or len(target_pts) < 3 or len(source_pts) != len(target_pts):
        return image.copy()
    
    # Get image dimensions
    h, w = image.shape[:2]
    is_gray = len(image.shape) == 2
    if is_gray:
        image = image[:, :, None]  # Add channel dimension for consistency
    
    # Convert points to float64 for numerical stability
    source_pts = source_pts.astype(np.float64)
    target_pts = target_pts.astype(np.float64)
    
    # Define RBF
    def rbf(r):
        return np.sqrt(r ** 2 + alpha ** 2)
    n = len(source_pts)
    
    # Build RBF matrix (pairwise distances between target points)
    r_matrix = np.sqrt(np.sum((target_pts[:, None] - target_pts[None, :]) ** 2, axis=2))
    A = rbf(r_matrix)
    A += np.eye(n) * eps  
    
    # Compute offsets
    b_x = source_pts[:, 0] - target_pts[:, 0]
    b_y = source_pts[:, 1] - target_pts[:, 1]
    
    # Solve linear system
    try:
        w_x = np.linalg.solve(A, b_x)
        w_y = np.linalg.solve(A, b_y)
    except np.linalg.LinAlgError:
        return image.copy()[:, :, 0] if is_gray else image.copy()
    
    # Create target image grid
    y_q, x_q = np.meshgrid(np.arange(h, dtype=np.float64),
                            np.arange(w, dtype=np.float64),
                            indexing='ij')
    q_coords = np.stack([x_q.ravel(), y_q.ravel()], axis=1)  # Shape (h*w, 2)
    
    # Compute source coordinates via RBF
    # Distance from each target pixel to each target control point
    dists = np.sqrt(np.sum((q_coords[:, None] - target_pts[None, :]) ** 2, axis=2))
    rbf_vals = rbf(dists)
    
    # Calculate source position: p = q + (w @ rbf_vals)
    dx = rbf_vals @ w_x
    dy = rbf_vals @ w_y
    x_p = q_coords[:, 0] + dx
    y_p = q_coords[:, 1] + dy
    
    x_p = np.clip(x_p, 0, w - 1)
    y_p = np.clip(y_p, 0, h - 1)
    
    # Bilinear Interpolation
    x0 = np.floor(x_p).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_p).astype(int)
    y1 = y0 + 1
    
    # Clip bounds
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    
    # Fractional offsets
    dx_frac = x_p - x0
    dy_frac = y_p - y0
    
    # Expand for channel broadcasting
    dx_frac = dx_frac[:, None]
    dy_frac = dy_frac[:, None]
    
    # Interpolate
    img_00 = image[y0, x0]
    img_01 = image[y0, x1]
    img_10 = image[y1, x0]
    img_11 = image[y1, x1]
    
    img_0 = img_00 * (1 - dx_frac) + img_01 * dx_frac
    img_1 = img_10 * (1 - dx_frac) + img_11 * dx_frac
    img_interp = img_0 * (1 - dy_frac) + img_1 * dy_frac
    
    # Reshape and restore
    warped_image = img_interp.reshape(h, w, -1)
    if is_gray:
        warped_image = warped_image[:, :, 0]
    
    return warped_image.astype(image.dtype)

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
