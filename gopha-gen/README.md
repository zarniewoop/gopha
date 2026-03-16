planned changes:

Balls Too Easy for FRC (Static Texture)
You’re correct.

Right now balls have:

Static color
Static lighting direction
Static internal texture (none)
Analytic motion with perfectly predictable velocity
For modern FRC, that’s easy:

Clean edges
Constant velocity
No internal motion
No rotation
Why This Is Easy for FRC
Most interpolation models rely heavily on:

Optical flow smoothness
Constant velocity assumptions
Texture matching
Your balls violate none of those assumptions.

✅ What Makes FRC Hard (But Still Aesthetic)
To stress FRC properly while keeping it visually pleasing:

A. Rotating Internal Texture
Give balls:

A radial pattern
A stripe pattern
A procedural swirl
A fractal fill
Then rotate texture independently of translation.

This breaks constant brightness constancy assumptions.

B. Specular Highlight Motion
Right now lighting is fixed.

If you:

Slowly move the light vector over time
Or make it orbit
Then specular highlight motion != object motion.

FRC models often break on that.

C. Micro-Detail Subpixel Texture
Add:

Fine procedural detail near Nyquist frequency
Rotating slightly
That stresses motion vector quantization and chroma subsampling.

D. Nonlinear Motion
Triangle wave = piecewise linear.

Instead:

Add slight sinusoidal acceleration
Add easing curves
Add quasi-periodic motion
That stresses interpolation models more than constant velocity.

A) Encoder stress
B) FRC testing
C) Broadcast chain compliance

That changes everything.

We want:

✅ Motion types
✅ Subpixel detail
✅ Clean chroma edges
✅ Known alignment anchors
✅ Controlled frequency sweeps
✅ Aesthetic structure

Not noise.

7️⃣ Suggested New Modes (High-Level)
Here’s what I recommend adding:

Mode 1: Fractal Zoom Plasma
Animated Mandelbrot / Julia zoom.

Why?

Structured detail
Multi-scale frequencies
High motion complexity
Visually beautiful
Great for encoder stress + FRC.

Mode 2: Rotating Patterned Discs
Balls replaced with:

Checkerboard discs
Radial spokes
Log-polar spirals
Fractal fills
With independent:

Translation
Rotation
Texture phase
Mode 3: Frequency Sweep Bars
Horizontally moving sinusoidal bars
Increasing spatial frequency over time.

Tests:

Motion vector limits
Nyquist interaction
Chroma bleed
Mode 4: Layered Parallax Scene
Multiple depth layers:

Slow background
Medium mid layer
Fast foreground
With transparency.

FRC struggles with layered motion.

Mode 5: Subpixel Oscillator Test
Objects oscillate:

[
x(t) = A \sin(2\pi f t)
]

Where amplitude < 1 pixel.

Brutal for FRC.

Big Strategic Suggestion
Instead of one monolithic mode:

Create:

--scene plasma
--scene fractal
--scene parallax
--scene sweep
--scene hybrid

Mutually exclusive base scene.

Then overlays:

balls
tickwheel
frequency grid
subpixel markers
9️⃣ Immediate Fixes I Strongly Recommend
✅ Fix ball lighting bug (nz[2])
✅ Increase anchor-snap to ~0.5/FPS
✅ Move plasma math to float64
✅ Add explicit chroma siting option
✅ Add rotating texture to balls

1️⃣ Architectural Direction
Core Principle
Separate:

Timebase engine (deterministic, anchor-safe)
Scene generator (pure function of time)
Overlays
Output pipeline (RGB → YUV)
Metadata contract
Everything must be:

Frame
=
𝑓
(
𝑡
,
seed
,
config
)
Frame=f(t,seed,config)
No hidden state. No frame-to-frame accumulation.

That guarantees:

50 vs 60 consistency
Deterministic regeneration
Easy canonical freezing
2️⃣ Timebase — Make It Bulletproof
Since per‑second FRC alignment is your primary concern:

✅ Proposed Time System
Instead of:

python
t = i / FPS

Run

Use:

python
frame_duration = 1.0 / FPS
t = i * frame_duration

Run

Then anchor snap with:

python
anchor_snap = 0.5 / FPS
anchor_period = 1.0

Run

This ensures:

Exact per-second alignment
Stable integer second frames
Identical analytic state at t=1.0, 2.0, 3.0 across 50/60
Optional (Even Cleaner)
Represent time internally as:

𝑡
=
𝑖
FPS
t= 
FPS
i
​
 
but compute motion phase as:

𝜙
=
𝑖
frames_per_cycle
ϕ= 
frames_per_cycle
i
​
 
Where:

frames_per_cycle
=
cycle_duration
×
FPS
frames_per_cycle=cycle_duration×FPS
This removes floating time entirely from motion phase calculations.

For FRC validation, this can eliminate subtle drift questions.

3️⃣ Ball System Redesign (Critical)
Right now balls are too “easy”.

Let’s redesign them as:

Procedural Disc Object
Each disc has:

Position(t)
Radius
Translation velocity
Rotation velocity
Texture generator
Shading model
✅ Internal Texture Options
Add:

--disc-texture
    flat
    checker
    radial
    spiral
    fractal
    noise-lowfreq
    noise-bandpass

Key: texture is function of polar coordinates inside disc.

Example:

Radial spokes:

𝑇
(
𝑟
,
𝜃
,
𝑡
)
=
sin
⁡
(
𝑁
𝜃
+
𝜔
𝑡
)
T(r,θ,t)=sin(Nθ+ωt)
Spiral:

𝑇
(
𝑟
,
𝜃
,
𝑡
)
=
sin
⁡
(
𝑁
𝜃
+
𝑘
𝑟
+
𝜔
𝑡
)
T(r,θ,t)=sin(Nθ+kr+ωt)
These are beautiful, structured, and brutal for FRC.

✅ Independent Rotation
Translation:

𝑥
(
𝑡
)
,
𝑦
(
𝑡
)
x(t),y(t)
Rotation:

𝜃
(
𝑡
)
θ(t)
Now optical flow inside the object is non-uniform.

That breaks simple motion interpolation assumptions.

✅ Specular Highlight Motion
Move light vector over time:

𝐿
(
𝑡
)
=
(
cos
⁡
(
𝜔
𝑡
)
,
sin
⁡
(
𝜔
𝑡
)
,
0.7
)
L(t)=(cos(ωt),sin(ωt),0.7)
Specular motion ≠ object motion.

Very revealing for frame synthesis.

4️⃣ Plasma Improvements
You noticed potential drift.

Two improvements:

✅ Remove Per-Frame Normalization
Instead of:

python
F01 = (F - F.min()) / (F.max() - F.min())

Run

Pre-compute theoretical bounds and normalize once globally.

Otherwise you introduce artificial temporal contrast modulation.

For FRC and encoder evaluation, that’s undesirable.

✅ Make Plasma Phase Rational
Instead of:

𝜔
𝑡
ωt
Use:

2
𝜋
𝑖
frames_per_cycle
2π 
frames_per_cycle
i
​
 
That ensures exact looping and cross-FPS consistency.

5️⃣ YUV / Colour — Fix It Properly
Since you are A+B+C focused:

This is important.

✅ Explicit Metadata Contract
Add to output:

Sidecar JSON:

json
{
  "range": "limited",
  "matrix": "bt709",
  "transfer": "bt709",
  "primaries": "bt709",
  "chroma_siting": "center"
}

Then document:

Encoder must match this.

✅ Add Switches
smalltalk
--range full|limited
--matrix bt601|bt709|bt2020
--transfer bt709|srgb|bt2020-10
--chroma-siting center|cosited

Broadcast chain compliance testing demands this.

6️⃣ Add Scene Types (Research Mode)
You want configurability but aesthetic value.

Here are high-value scenes:

Scene: fractal_zoom
Animated Mandelbrot or Julia.

Smooth zoom
Palette cycling
Infinite detail
No random noise
Visually beautiful
Very hard for encoders.

Scene: parallax_layers
3–5 depth planes:

Background fractal
Mid geometric pattern
Foreground rotating discs
Alpha blending
Different velocities per layer.

FRC hates occlusion boundaries.

Scene: frequency_sweep
Horizontal bars:

𝐼
(
𝑥
,
𝑡
)
=
sin
⁡
(
2
𝜋
𝑓
(
𝑡
)
𝑥
)
I(x,t)=sin(2πf(t)x)
Where f(t) increases over time.

This lets you measure:

When motion estimation breaks
When chroma smears
When scaling fails
Scene: subpixel_probe
Small features oscillating:

𝑥
(
𝑡
)
=
𝐴
sin
⁡
(
2
𝜋
𝑓
𝑡
)
x(t)=Asin(2πft)
Where A < 1 pixel.

Excellent for detecting motion quantization and FRC rounding.

7️⃣ Canonical Set Strategy (For Your Teams)
Once generator is mature:

Define something like:

Canonical Suite v1.0
SEQ-A1: Plasma + Rotating Discs (medium motion)
SEQ-A2: High-frequency fractal zoom
SEQ-A3: Parallax multi-layer stress
SEQ-A4: Subpixel oscillation stress
SEQ-A5: Colour compliance pattern (range/matrix check)
Each with:

Fixed seed
Fixed resolution
Fixed FPS (50 and 60 versions)
Fixed duration (e.g., 30s)
Documented metadata
Now you have:

Internal gold standard
Vendor comparison baseline
Repeatable regression test content
8️⃣ 64-bit Recommendation
Yes.

For analytic calculations:

Use float64 everywhere until final RGB clamp.
Convert to float32 only if needed for memory.
This reduces tiny phase inconsistencies.

For your use case (A+B+C), float64 internal math is justified.

9️⃣ Big Strategic Suggestion
Add:

--determinism-level
    strict
    fast

Strict:

float64
rational phase
anchor snap enabled
Fast:

current behaviour
That gives you research flexibility without performance pain.

