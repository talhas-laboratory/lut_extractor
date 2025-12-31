To build **Lattice.ai**, you must execute a "Functional Friction" design strategy. Most modern web apps try to be invisible (fluid, soft, round). Lattice.ai should be **felt**. It must feel like a piece of industrial equipment that has been digitised—heavy, reliable, and precise.

Here is the blueprint for a **Neo-Brutalist Organic** frontend.

---

## 1. The Design System: "Raw Precision"

### The Foundation (Neo-Brutalist)

* **The "Shadow of Weight":** Do not use `box-shadow` with blur. Use hard offsets.
* *Rule:* All primary containers must have `box-shadow: 8px 8px 0px 0px #000000;`.
* *Purpose:* This simulates physical depth. When a button is "clicked," you animate it to `4px` or `0px`, creating a tactile "clunk" sensation.


* **The Grid:** Use a visible 24px or 32px grid background on the workspace.
* *Purpose:* It reinforces the idea that the user is working within a mathematical coordinate system, not an "art canvas."



### The "Organic" Counter-Point

To prevent the app from feeling cold or "dead," we introduce organic movement through **physics-based micro-interactions** and **color theory**.

* **Wobbly Dividers:** Instead of perfectly straight lines for section breaks, use SVGs with a very slight "hand-drawn" or "oscillating" path.
* *Purpose:* It signals that while the math is rigid (Lattice), the source material (Film/Light) is organic and fluid.


* **Grain & Noise:** Apply a global CSS `noise` overlay at 2% opacity.
* *Purpose:* Digital color science often looks "sterile." A subtle grain makes the UI feel like it exists on a physical film stock.



---

## 2. Component-Level Logic

### A. The "Trinity" Upload Nodes

* **Design:** Each upload zone is a "Node." When a file is dropped, a "Power Line" (a thick, 4px black line) should animate and connect to the central "Engine."
* **Purpose:** To visually illustrate the **dependency**. The engine cannot run without all three mathematical inputs. It turns a boring form into a "Machine Assembly."

### B. The 3D Lattice Visualizer (The Hero)

* **Tech:** Use `Three.js` or `React Three Fiber`.
* **Design:** Render the 33x33x33 color cube as a point cloud.
* **Interaction:** When the user adjusts "Skin Tone Protection," the points in the "Skin" coordinate area of the cube should glow or remain stationary while the others shift.
* **Purpose:** This is the **"Receipt of Work."** If a user pays $5, they need to see the "Math" they just bought. Seeing the points move in 3D space justifies the price tag better than a loading bar.

### C. Typography Pairings

| Use Case | Font Type | Choice (Example) |
| --- | --- | --- |
| **Headlines** | Heavy Grotesque | *Integral CF* or *Oswald* (Bold 900) |
| **Data/Logs** | Monospace | *JetBrains Mono* or *Roboto Mono* |
| **Body/Label** | Clean Sans | *Inter* (Semi-bold) |

---

## 3. Technical Execution Strategy

### The CSS Stack: Tailwind + Framer Motion

Framer Motion is essential for the "Organic" feel. Use it to create "Spring" animations rather than "Ease" animations.

```javascript
// Example of a "Brutal" but "Organic" Button spring
const buttonVariants = {
  hover: { x: -2, y: -2, boxShadow: "10px 10px 0px 0px #000" },
  tap: { x: 8, y: 8, boxShadow: "0px 0px 0px 0px #000" }
};

```

### The "Autopsy" Loading Sequence

When the user clicks "Extract DNA," do not show a spinner. Show a **Log Stream**.

1. **Step 1:** Scan the Reference (Visual: Vertical line pass).
2. **Step 2:** Isolate Luminance (Visual: The image turns black and white).
3. **Step 3:** Generate TPS Map (Visual: A wireframe mesh grows over the image).

* **Purpose:** This educates the user on *why* this is better than a preset. It’s an "Autopsy" of their image.

---

## 4. The "Impulse" Conversion Flow (UX)

To support the $5/extraction model, the friction must be removed at the end, not the beginning.

1. **The Hook:** Allow the user to upload and see the **3D Lattice Animation** for free.
2. **The Paywall:** Only when they click "Download .CUBE" do you trigger the payment.
3. **The "Bridge":** Since users discover this on mobile but use it on desktop, the "Email to Workstation" button should be the largest, most vibrant element after payment.

---

## 5. Visual "Easter Eggs"

* **Cursor:** Replace the standard cursor with a "Crosshair" or a "Color Picker" icon within the workspace.
* **Hover States:** When hovering over data points, use a "Magnifying Glass" effect that shows the raw Hex code and CIELAB coordinates.
