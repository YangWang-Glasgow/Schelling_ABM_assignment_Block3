import numpy as np
import random
import os
import base64
import io
import json
from PIL import Image as PilImage, ImageDraw

# ── Model ─────────────────────────────────────────────────────────────────────

class SchellingModel:

    def __init__(self, width=20, height=20, density=0.9, similarity_threshold=0.5):
        self.width = width
        self.height = height
        self.density = density
        self.similarity_threshold = similarity_threshold

        self.EMPTY   = 0
        self.GROUP_A = 1
        self.GROUP_B = 2

        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.init_agents()

    def init_agents(self):
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.density:
                    self.grid[y, x] = random.choice([self.GROUP_A, self.GROUP_B])
                else:
                    self.grid[y, x] = self.EMPTY

    def get_neighbors(self, x, y):
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    n = self.grid[ny, nx]
                    if n != self.EMPTY:
                        neighbors.append(n)
        return neighbors

    def is_happy(self, x, y):
        agent = self.grid[y, x]
        if agent == self.EMPTY:
            return True
        neighbors = self.get_neighbors(x, y)
        if not neighbors:
            return True
        similar = sum(1 for n in neighbors if n == agent)
        return (similar / len(neighbors)) >= self.similarity_threshold

    def step(self):
        unhappy_agents = [
            (x, y) for y in range(self.height)
            for x in range(self.width)
            if self.grid[y, x] != self.EMPTY and not self.is_happy(x, y)
        ]
        random.shuffle(unhappy_agents)

        empty_spaces = [
            (x, y) for y in range(self.height)
            for x in range(self.width)
            if self.grid[y, x] == self.EMPTY
        ]
        random.shuffle(empty_spaces)

        for (x, y) in unhappy_agents:
            if not empty_spaces:
                break
            new_x, new_y = empty_spaces.pop()
            self.grid[new_y, new_x] = self.grid[y, x]
            self.grid[y, x] = self.EMPTY
            empty_spaces.append((x, y))  # vacated cell becomes available

    def percent_unhappy(self):
        agents = [
            (x, y) for y in range(self.height)
            for x in range(self.width)
            if self.grid[y, x] != self.EMPTY
        ]
        if not agents:
            return 0.0
        unhappy = sum(1 for x, y in agents if not self.is_happy(x, y))
        return 100 * unhappy / len(agents)

    def percent_similar(self):
        agents = [
            (x, y) for y in range(self.height)
            for x in range(self.width)
            if self.grid[y, x] != self.EMPTY
        ]
        if not agents:
            return 0.0
        total_similar = 0
        total_neighbors = 0
        for x, y in agents:
            neighbors = self.get_neighbors(x, y)
            total_similar += sum(1 for n in neighbors if n == self.grid[y, x])
            total_neighbors += len(neighbors)
        return 100 * total_similar / total_neighbors if total_neighbors > 0 else 100.0


# ── Rendering (pure Pillow, zero matplotlib) ──────────────────────────────────

CELL = 8  # pixels per grid cell

COLOUR_EMPTY   = (230, 230, 230)
COLOUR_GROUP_A = (50,  100, 220)   # blue
COLOUR_GROUP_B = (230, 120,  30)   # orange
COLOUR_BG      = (255, 255, 255)


def render_grid(model, step_num, pct_similar, pct_unhappy):
    """Render the Schelling grid as a Pillow image with a header bar."""
    header_h = 28
    img_w = model.width  * CELL
    img_h = model.height * CELL + header_h

    img  = PilImage.new("RGB", (img_w, img_h), COLOUR_BG)
    draw = ImageDraw.Draw(img)

    # Header text
    header = f"Step {step_num}   |   % Similar: {pct_similar:.1f}   |   % Unhappy: {pct_unhappy:.1f}"
    draw.text((6, 6), header, fill=(60, 60, 60))

    # Grid cells
    colour_map = {0: COLOUR_EMPTY, 1: COLOUR_GROUP_A, 2: COLOUR_GROUP_B}
    for y in range(model.height):
        for x in range(model.width):
            colour = colour_map[model.grid[y, x]]
            x0 = x * CELL
            y0 = y * CELL + header_h
            draw.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], fill=colour)

    return img


def render_plot(similar_vals, unhappy_vals, steps, plot_w=380, plot_h=None):
    """Render the segregation statistics chart as a Pillow image."""
    if plot_h is None:
        plot_h = plot_w  # square by default

    pad_l, pad_r, pad_t, pad_b = 52, 16, 32, 40
    img  = PilImage.new("RGB", (plot_w, plot_h), COLOUR_BG)
    draw = ImageDraw.Draw(img)

    # Background
    draw.rectangle([0, 0, plot_w - 1, plot_h - 1], outline=(200, 200, 200))

    # Title
    draw.text((plot_w // 2 - 60, 8), "Segregation Statistics", fill=(50, 50, 50))

    # Legend
    draw.ellipse([pad_l, 20, pad_l + 8, 28], fill=(34, 150, 60))
    draw.text((pad_l + 11, 19), "% Similar", fill=(34, 150, 60))
    draw.ellipse([pad_l + 80, 20, pad_l + 88, 28], fill=(200, 40, 40))
    draw.text((pad_l + 91, 19), "% Unhappy", fill=(200, 40, 40))

    # Axes
    ax_x0 = pad_l
    ax_x1 = plot_w - pad_r
    ax_y0 = pad_t
    ax_y1 = plot_h - pad_b
    draw.line([ax_x0, ax_y0, ax_x0, ax_y1], fill=(160, 160, 160), width=1)
    draw.line([ax_x0, ax_y1, ax_x1, ax_y1], fill=(160, 160, 160), width=1)

    # Y-axis ticks (0–100%)
    for pct in range(0, 101, 20):
        py = ax_y1 - int(pct / 100 * (ax_y1 - ax_y0))
        draw.line([ax_x0 - 4, py, ax_x0, py], fill=(160, 160, 160))
        draw.text((2, py - 6), str(pct), fill=(120, 120, 120))

    # X-axis label
    draw.text((ax_x0 + (ax_x1 - ax_x0) // 2 - 12, plot_h - 14), "Step", fill=(100, 100, 100))

    def to_px(step, val):
        sx = steps if steps > 0 else 1
        x = ax_x0 + int((step / sx) * (ax_x1 - ax_x0))
        y = ax_y1 - int((val / 100) * (ax_y1 - ax_y0))
        return x, y

    n = len(similar_vals)
    if n > 1:
        for i in range(1, n):
            draw.line([to_px(i - 1, similar_vals[i - 1]), to_px(i, similar_vals[i])],
                      fill=(34, 150, 60), width=2)
            draw.line([to_px(i - 1, unhappy_vals[i - 1]), to_px(i, unhappy_vals[i])],
                      fill=(200, 40, 40), width=2)

    return img


def combine_frames(grid_img, plot_img):
    """Stitch the grid and plot side by side on a neutral background."""
    gw, gh = grid_img.size
    pw, ph = plot_img.size
    h = max(gh, ph)
    combined = PilImage.new("RGB", (gw + pw + 12, h), (240, 240, 240))
    combined.paste(grid_img, (0, (h - gh) // 2))
    combined.paste(plot_img, (gw + 12, (h - ph) // 2))
    return combined


def frame_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Schelling Segregation Model</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', sans-serif;
    background: #1e1e2e; color: #cdd6f4;
    display: flex; flex-direction: column; align-items: center;
    padding: 28px 16px;
  }}
  h1 {{ font-size: 1.3rem; margin-bottom: 4px; }}
  .subtitle {{ font-size: 0.78rem; color: #888; margin-bottom: 18px; text-align: center; }}

  #display {{
    border-radius: 8px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.55);
    max-width: 100%;
  }}

  .controls {{
    display: flex; gap: 10px; align-items: center;
    margin-top: 14px; flex-wrap: wrap; justify-content: center;
  }}
  button {{
    padding: 7px 16px; border-radius: 6px; border: none; cursor: pointer;
    font-size: 0.82rem; font-weight: 600;
    background: #313244; color: #cdd6f4; transition: background 0.2s;
  }}
  button:hover {{ background: #45475a; }}
  #playBtn {{ background: #89b4fa; color: #1e1e2e; }}
  #playBtn:hover {{ background: #74c7ec; }}

  label {{ font-size: 0.8rem; color: #a6adc8; display: flex; align-items: center; gap: 6px; }}
  input[type=range] {{ accent-color: #89b4fa; width: 120px; }}

  .legend {{
    display: flex; gap: 18px; margin-top: 12px; font-size: 0.78rem; color: #a6adc8;
  }}
  .dot {{
    width: 12px; height: 12px; border-radius: 50%; display: inline-block;
    margin-right: 5px; vertical-align: middle;
  }}
  .info {{ font-size: 0.75rem; color: #6c7086; margin-top: 8px; }}
</style>
</head>
<body>

<h1>🏘️ Schelling Segregation Model</h1>
<p class="subtitle">
  {steps} steps &nbsp;·&nbsp; {width}×{height} grid &nbsp;·&nbsp;
  density = {density} &nbsp;·&nbsp; threshold = {threshold}
</p>

<img id="display" src="" alt="simulation frame">

<div class="legend">
  <span><span class="dot" style="background:#3264dc"></span>Group A</span>
  <span><span class="dot" style="background:#e6781e"></span>Group B</span>
  <span><span class="dot" style="background:#e6e6e6; border:1px solid #aaa"></span>Empty</span>
</div>

<div class="controls">
  <button id="playBtn" onclick="togglePlay()">⏸ Pause</button>
  <button onclick="stepBack()">◀ Step</button>
  <button onclick="stepFwd()">Step ▶</button>
  <button onclick="restart()">↺ Restart</button>
  <label>
    Speed
    <input type="range" id="speed" min="100" max="1500" value="800" oninput="updateSpeed()">
  </label>
</div>
<div class="info" id="info">Step 0 / {steps}</div>

<script>
const frames = {frames_json};
let idx = 0, playing = true, timer = null;
let interval = 800;

function show(i) {{
  document.getElementById("display").src = "data:image/png;base64," + frames[i];
  document.getElementById("info").textContent = "Step " + i + " / {steps}";
}}

function next() {{
  if (idx < frames.length - 1) {{
    idx++; show(idx);
  }} else {{
    clearInterval(timer); playing = false;
    document.getElementById("playBtn").textContent = "▶ Play";
  }}
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "⏸ Pause" : "▶ Play";
  if (playing) timer = setInterval(next, interval);
  else clearInterval(timer);
}}

function stepFwd()  {{ if (idx < frames.length - 1) {{ idx++; show(idx); }} }}
function stepBack() {{ if (idx > 0) {{ idx--; show(idx); }} }}

function restart() {{
  clearInterval(timer); idx = 0; show(0);
  playing = true;
  document.getElementById("playBtn").textContent = "⏸ Pause";
  timer = setInterval(next, interval);
}}

function updateSpeed() {{
  interval = 1600 - parseInt(document.getElementById("speed").value);
  if (playing) {{ clearInterval(timer); timer = setInterval(next, interval); }}
}}

show(0);
timer = setInterval(next, interval);
</script>
</body>
</html>
"""


# ── Main entry point ──────────────────────────────────────────────────────────

def run_schelling_model(
    steps: int = 30,
    width: int = 50,
    height: int = 50,
    density: float = 0.9,
    similarity_threshold: float = 0.5
):
    """
    Runs the Schelling Segregation ABM and saves a self-contained interactive
    HTML file (schelling.html) to the current working directory.

    Parameters
    ----------
    steps : int
        Number of simulation steps.
    width, height : int
        Dimensions of the spatial grid.
    density : float
        Proportion of grid cells initially occupied by agents.
    similarity_threshold : float
        Required proportion of similar neighbours for an agent to be happy.

    Returns
    -------
    str
        Path to the saved HTML file.
    """
    model = SchellingModel(
        width=width, height=height,
        density=density, similarity_threshold=similarity_threshold
    )

    similar_vals, unhappy_vals = [], []
    frame_b64s = []

    # Match plot height to grid height for a tidy side-by-side layout
    grid_px_h = height * CELL + 28   # 28 = header bar
    grid_px_w = width  * CELL

    print(f"Running Schelling model: {steps} steps on a {width}×{height} grid...")

    for step_num in range(steps):
        model.step()

        pct_sim = model.percent_similar()
        pct_unh = model.percent_unhappy()
        similar_vals.append(pct_sim)
        unhappy_vals.append(pct_unh)

        grid_img = render_grid(model, step_num + 1, pct_sim, pct_unh)
        plot_img  = render_plot(similar_vals, unhappy_vals, steps,
                                plot_w=grid_px_w, plot_h=grid_px_h)
        combined  = combine_frames(grid_img, plot_img)
        frame_b64s.append(frame_to_b64(combined))

        if (step_num + 1) % 10 == 0 or step_num == 0:
            print(f"  Step {step_num + 1:3d}: {pct_sim:.1f}% similar, {pct_unh:.1f}% unhappy")

    print("Rendering complete. Writing HTML...")

    html = HTML_TEMPLATE.format(
        steps=steps,
        width=width,
        height=height,
        density=density,
        threshold=similarity_threshold,
        frames_json=json.dumps(frame_b64s),
    )

    save_path = os.path.join(os.getcwd(), "schelling.html")
    with open(save_path, "w") as f:
        f.write(html)

    print(f"Saved → {save_path}")
    return save_path


# Keep the no-graphs variant for compatibility — it now just calls the main runner
def run_schelling_model_no_graphs(
    steps: int = 30,
    width: int = 50,
    height: int = 50,
    density: float = 0.9,
    similarity_threshold: float = 0.5
):
    """Alias that runs the full model. The no-graphs variant is deprecated."""
    return run_schelling_model(
        steps=steps, width=width, height=height,
        density=density, similarity_threshold=similarity_threshold
    )


if __name__ == "__main__":
    run_schelling_model(
        steps=30,
        width=50,
        height=50,
        density=0.9,
        similarity_threshold=0.5
    )