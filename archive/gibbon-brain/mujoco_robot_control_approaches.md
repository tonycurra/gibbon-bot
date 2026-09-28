# MuJoCo Robot Control Approaches
## High-Level Architectures for Autonomous Text-Controlled Robots

---

## **📌 Overview**

This document outlines **high-level architectural approaches** for building an autonomous robot control system that:
- Accepts **natural language commands** (e.g., "raise your right arm", "climb the pole")
- Operates in **MuJoCo simulation**
- Provides **text feedback**
- Can be deployed on **edge hardware** (Hailo-10H, Intel 358H)

Each approach balances **complexity, capability, and hardware feasibility**. Choose based on your priorities:
- **Speed of development** → Rule-based
- **Adaptability** → Learning-based (RL/IL)
- **Generalization** → Vision-Language-Action (VLA)
- **Scalability** → Multi-agent

---

---

## **🏗️ Architecture Categories**

### **1️⃣ Rule-Based + Direct Control**
*Simplest approach: Predefined mappings from text to actions.*

#### **Architecture**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Text     │────▶│   Command Parser│────▶│  Joint Control  │
│  (e.g. "raise   │     │ (Rule-Based/DSL)│     │ (Position/Torque│
│    right arm")  │     └─────────────────┘     │    Targets)      │
└─────────────────┘                           └─────────────────┘
                                                      │
                                                      ▼
                                               ┌─────────────────┐
                                               │   MuJoCo Sim    │
                                               │   (Robot Model) │
                                               └─────────────────┘
```

#### **Variants**
| Variant | Description | Data Needed |
|---------|-------------|-------------|
| **Text → Joint Trajectories** | Hardcoded joint angles for each command | None (manual) |
| **Text → Inverse Kinematics (IK)** | Compute joint angles to reach targets (hardcoded or vision-based) | None (or vision for targets) |

#### **Pros**
✅ **Zero training** – Works out of the box
✅ **Deterministic & debuggable** – No black boxes
✅ **Ultra-low latency** – Microseconds per command
✅ **Runs anywhere** – Even on Raspberry Pi

#### **Cons**
❌ **No adaptability** – Fails on unseen scenarios
❌ **No vision** – Can't react to dynamic environments
❌ **Manual effort** – Requires hand-tuning for each command

#### **Use Cases**
- Rapid prototyping
- Simple demos
- Fixed-environment tasks (e.g., factory arms)

#### **Hardware Fit**
| Hardware | Fit | Notes |
|----------|-----|-------|
| **CPU** | ✅ Perfect | No acceleration needed |
| **Hailo-10H** | ✅ Overkill | Runs effortlessly |
| **Intel 358H** | ✅ Overkill | Runs effortlessly |

---

### **2️⃣ Learning-Based (Reinforcement Learning / Imitation Learning)**
*Train policies to map observations → actions.*

#### **Architecture**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Text     │────▶│  Policy Selector │────▶│   RL/IL Policy  │
│  (e.g. "walk")   │     │  (Text → Policy) │     │ (Neural Network) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                      │
                                      ┌────────────────┴────────────────┐
                                      │                                 │
                  ┌───────────────────▼─────────────────▼────────────┐
                  │                   MuJoCo Sim (Observations)         │
                  │  [Joint Angles, Velocities, End-Effector Positions]  │
                  └───────────────────▲───────────────────────────────┘
                                      │
                                      └─────────────────────────────────┘
```

#### **Variants**
| Variant | Input | Training Method | Data Needed |
|---------|-------|-----------------|-------------|
| **Proprioception RL** | Joint states, velocities | PPO, SAC | Simulation (1M+ steps) |
| **Vision + Proprioception RL** | RGB/Depth + joint states | PPO, SAC | Simulation (10M+ steps) |
| **Imitation Learning (IL)** | Observations | Behavior Cloning | Demonstrations (teleop/IK) |

#### **Pros**
✅ **Adapts to dynamics** – Handles slipping, uneven terrain
✅ **No manual tuning** – Learns from data
✅ **Generalizes within task** – E.g., walking on different surfaces

#### **Cons**
❌ **Training required** – Needs simulation or demos
❌ **Sample-inefficient** (RL) – Millions of steps for complex tasks
❌ **No language understanding** – Needs a separate text→policy mapper

#### **Use Cases**
- Dynamic movement (walking, climbing)
- Contact-rich tasks (pushing, grasping)
- Adaptive control (e.g., recovering from falls)

#### **Hardware Fit**
| Hardware | Fit | Notes |
|----------|-----|-------|
| **CPU** | ⚠️ Training: No | Inference only (small policies) |
| **GPU** | ✅ Training: Yes | Required for RL training |
| **Hailo-10H** | ✅ Inference | <10M param policies (INT8) |
| **Intel 358H** | ✅ Inference | <50M param policies (INT8) |

---

### **3️⃣ Vision-Language-Action (VLA) Models**
*End-to-end or modular systems that map **text + vision → actions**.*

#### **Architecture Options**

##### **A. End-to-End VLA (RT-2 Style)**
```
┌─────────────────┐     ┌─────────────────┐
│   User Text     │────▶│                 │
│  (e.g. "climb   │     │  VLA Model      │────▶┐
│    the pole")   │────▶│ (Transformer)   │    ┌▼───────────────┐
└─────────────────┘     │                 │    │ MuJoCo Actions │
┌─────────────────┐     │  [Text + Vision │    │ (Joint Torques)│
│   Camera Input  │────▶│   Encoders]     │    └───────────────┘
│  (RGB/Depth)    │     └─────────────────┘
└─────────────────┘
```

##### **B. Modular VLA (SayCan Style)**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Text     │────▶│   LLM (Value    │────▶│   Skill        │
│  (e.g. "climb   │     │    Judge)       │     │  Policies       │
│    the pole")   │     │  "Is this      │────▶│  (RL/IL Trained)│────▶┐
└─────────────────┘     │   possible?"     │     └─────────────────┘    ┌▼───────────────┐
                        └─────────────────┘                          │ MuJoCo Actions │
                              │                                     └───────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Vision Model                              │
│  (Object Detection, Pose Estimation, Scene Understanding)        │
└──────────────────────────────────────────────────────────────────┘
```

#### **Variants**
| Variant | Description | Data Needed | Language Capability |
|---------|-------------|-------------|----------------------|
| **End-to-End VLA** | Single model for text+vision→action | Large-scale (text, vision, action) | Full NLU |
| **Modular VLA (SayCan)** | LLM + Vision + Skill Policies | Separate datasets per module | High-level commands |
| **Diffusion Policies** | Treat actions as sequence generation | Large-scale trajectory data | Medium |

#### **Pros**
✅ **Generalizes to new commands** – Understands natural language
✅ **Vision-aware** – Reacts to environment changes
✅ **Unified model** (End-to-End) – No explicit parsing/planning

#### **Cons**
❌ **Data-hungry** – Needs large datasets (or strong pretraining)
❌ **Compute-heavy** – Hard to fit on edge
❌ **Debugging is hard** – Black-box behavior

#### **Use Cases**
- General-purpose robot control
- Complex, multi-step tasks (e.g., "find the pole, climb it, then wave")
- Human-robot interaction

#### **Hardware Fit**
| Model Size | Hailo-10H | Intel 358H | Notes |
|------------|-----------|------------|-------|
| **<10M params** | ✅ Yes | ✅ Yes | TinyBERT + MobileNetV3 |
| **10-50M params** | ❌ No | ✅ Yes | DistilBERT + EfficientNet |
| **50M-100M params** | ❌ No | ⚠️ Maybe | Quantization required |
| **>100M params** | ❌ No | ❌ No | Needs GPU/Cloud |

---

### **4️⃣ Multi-Agent Orchestration**
*Split responsibilities across specialized agents for scalability and hardware efficiency.*

#### **Architecture**
```
┌─────────────────┐     ┌─────────────────┐
│   User Text     │────▶│  Language Agent │
│                 │     │  (LLM/Parser)   │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                        Vision Agent                               │
│  (Scene Understanding: Objects, Poses, Affordances)               │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Control Agent                               │
│  (Low-Level Execution: Joint Torques/Positions)                  │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   MuJoCo Sim    │
               └─────────────────┘
```

#### **Agent Specializations**
| Agent | Role | Input | Output | Hardware |
|-------|------|-------|--------|----------|
| **Vision** | Scene understanding | RGB/Depth | Object poses, affordances | Intel 358H |
| **Language** | Text parsing | Text + Scene | High-level goals (e.g., "climb pole") | Intel 358H |
| **Control** | Low-level execution | Goals + Sensor Data | Joint torques/positions | Hailo-10H |

#### **Pros**
✅ **Modular** – Easy to upgrade/debug individual components
✅ **Hardware-efficient** – Match agent size to chip capabilities
✅ **Parallelizable** – Agents can run on separate cores/NPUs
✅ **Scalable** – Add more agents as needed (e.g., memory, planning)

#### **Cons**
❌ **Complexity** – Requires inter-agent communication
❌ **Latency** – Communication overhead between agents
❌ **Synchronization** – Need to manage shared state

#### **Use Cases**
- Real-time control with hardware constraints
- Large-scale systems (e.g., multiple robots)
- Hybrid intelligence (symbolic + learned components)

#### **Hardware Fit**
- **Vision Agent** → **Intel 358H** (handles CNNs)
- **Language Agent** → **Intel 358H** (handles LLMs)
- **Control Agent** → **Hailo-10H** (low-power, real-time)

---

---

## **📊 Comparison Table**

| **Approach** | **Complexity** | **Training Needed** | **Vision** | **Language** | **Adaptability** | **Hardware Fit** | **Best For** |
|--------------|---------------|--------------------|------------|--------------|------------------|-----------------|--------------|
| Rule-Based + IK | ⭐ | ❌ No | ❌ No | ⚠️ Basic | ❌ Low | CPU/Hailo/Intel | Prototyping, fixed tasks |
| RL (Proprioception) | ⭐⭐ | ✅ Yes (RL) | ❌ No | ❌ No | ✅ Medium | CPU (infer), GPU (train) | Dynamic movement |
| RL (Vision + Proprio) | ⭐⭐⭐ | ✅ Yes (RL) | ✅ Yes | ❌ No | ✅ High | GPU (train), Intel (infer) | Vision-guided tasks |
| Imitation Learning | ⭐⭐ | ✅ Yes (IL) | ✅ Optional | ❌ No | ✅ Medium | GPU (train), Intel/Hailo (infer) | Data-efficient learning |
| Modular VLA (SayCan) | ⭐⭐⭐⭐ | ✅ Yes (IL/RL) | ✅ Yes | ✅ Yes | ✅ High | Intel 358H | Interpretable, flexible |
| End-to-End VLA (RT-2) | ⭐⭐⭐⭐⭐ | ✅ Yes (IL) | ✅ Yes | ✅ Yes | ✅ Very High | Intel 358H | General-purpose |
| Multi-Agent | ⭐⭐⭐⭐ | ⚠️ Optional | ✅ Yes | ✅ Yes | ✅ High | Hailo + Intel | Scalable, real-time |

---

---

## **🎯 Recommendations by Use Case**

### **🚀 Fastest to Implement**
**→ Rule-Based + IK**
- **Why?** No training, no data collection, works immediately.
- **When?** Prototyping, demos, fixed environments.
- **Next Step:** Add vision for target detection (IK → vision-based targets).

### **🤖 Learning Dynamics**
**→ RL (Proprioception or Vision + Proprioception)**
- **Why?** Adapts to physics, handles disturbances.
- **When?** Walking, climbing, contact-rich tasks.
- **Tools:** Stable Baselines3, Isaac Gym.

### **🗣️ Natural Language Understanding**
**→ Modular VLA (SayCan)**
- **Why?** Combines LLM flexibility with learned skills.
- **When?** Need to interpret complex commands.
- **Hardware:** Intel 358H.

### **🏗️ Scalable & Hardware-Efficient**
**→ Multi-Agent Orchestration**
- **Why?** Distribute workload across specialized chips.
- **When?** Edge deployment with hardware constraints.
- **Hardware:** Hailo-10H (control) + Intel 358H (vision + language).

### **🌌 Maximum Generalization**
**→ End-to-End VLA (RT-2 Style)**
- **Why?** Single model for all tasks.
- **When?** Research, general-purpose robots.
- **Hardware:** Intel 358H (quantized small model) or Cloud.

---

---

## **📱 Edge Hardware Guidelines**

### **Hailo-10H**
- **TOPS:** 26 (INT8)
- **Power:** 5-10W
- **Best For:**
  - Low-latency **control** (PD control, small RL policies)
  - **Vision models** <10M params (MobileNetV3, EfficientNet-B0)
  - **Text models** <10M params (TinyBERT, MobileBERT)
- **Limitations:**
  - Struggles with **>50M params**
  - No **training support** (inference only)

### **Intel 358H**
- **TOPS:** 48 (INT8)
- **Power:** 15-25W
- **Best For:**
  - **Vision-Language models** (10-50M params)
  - **RL policies** (10-50M params)
  - **LLMs** (25-66M params, quantized)
- **Limitations:**
  - Higher power consumption
  - Still limited for **>100M params**

### **Model Recommendations for Edge**
| Task | Recommended Model | Params | Hailo-10H | Intel 358H |
|------|-------------------|--------|-----------|------------|
| **Text Encoding** | TinyBERT | 14M | ✅ | ✅ |
| **Text Encoding** | MobileBERT | 25M | ❌ | ✅ |
| **Vision Encoding** | MobileNetV3-Small | 2.5M | ✅ | ✅ |
| **Vision Encoding** | EfficientNet-B0 | 5.3M | ✅ | ✅ |
| **VLA (Fused)** | TinyBERT + MobileNetV3 | ~10M | ✅ | ✅ |
| **RL Policy** | 2-layer MLP | 0.1M | ✅ | ✅ |
| **RL Policy** | Small CNN + MLP | 5M | ✅ | ✅ |

---

## **🤖 Small Model Recommendations for Orchestration**

### **Agent-Specific Tiny Models (All <50M params, Edge-Ready)**

#### **👁️ Vision Agents**
| Model | Params | Task | Top-1 Acc | Latency (Hailo) | Latency (Intel) | Best For |
|-------|--------|------|-----------|-----------------|------------------|----------|
| **MobileNetV3-Small** | 2.5M | Object Detection / Classification | ~70% (ImageNet) | ~10ms | ~5ms | General scene understanding |
| **MobileNetV3-Large** | 5.4M | Object Detection / Classification | ~75% (ImageNet) | ~15ms | ~8ms | Higher accuracy vision |
| **EfficientNet-B0** | 5.3M | Classification / Feature Extraction | ~77% (ImageNet) | ~15ms | ~8ms | Balanced accuracy/speed |
| **ShuffleNetV2** | 3.5M | Lightweight Vision | ~69% (ImageNet) | ~12ms | ~6ms | Extremely lightweight |
| **Yolo-Nano** | 1.9M | Real-Time Object Detection | ~50% (COCO mAP) | ~20ms | ~10ms | Pole/obstacle detection |
| **Depth Anything (Tiny)** | ~3M | Depth Estimation | - | ~25ms | ~12ms | 3D scene understanding |

#### **🗣️ Language Agents**
| Model | Params | Task | Hardware Fit | Best For |
|-------|--------|------|--------------|----------|
| **TinyBERT (4-layer)** | 14M | Text Encoding / Intent Classification | ✅ Both | Command parsing, simple NLU |
| **MobileBERT** | 25M | Text Encoding | ❌ Hailo / ✅ Intel | Better NLU, still lightweight |
| **DistilBERT** | 66M | Text Encoding | ❌ Hailo / ⚠️ Intel (quantized) | Complex commands, reasoning |
| **ALBERT-Tiny** | 11M | Text Encoding | ✅ Both | Very lightweight NLU |
| **BART-Tiny (Encoder)** | 14M | Text → Structured Output | ✅ Both | Command to action JSON |

#### **🎮 Control Agents (RL/IL Policies)**
| Model | Params | Input | Output | Training | Hardware Fit | Best For |
|-------|--------|-------|--------|----------|--------------|----------|
| **2-layer MLP** | 0.1M | Proprioception | Joint Torques | RL (PPO/SAC) | ✅ Both | Simple control tasks |
| **3-layer MLP** | 0.5M | Proprioception | Joint Positions | RL/IL | ✅ Both | Walking, basic locomotion |
| **Small CNN + MLP** | 2-5M | Vision + Proprio | Joint Torques | RL | ✅ Both | Vision-guided reaching |
| **GRU/ LSTM** | 1-3M | Temporal States | Actions | IL | ✅ Both | Sequential tasks |
| **Tiny Transformer** | 3-8M | State History | Actions | IL | ✅ Both | Memory-augmented control |

#### **🎯 Fused VLA Models (Single Model for Text+Vision→Action)**
| Model | Params | Vision Backbone | Text Backbone | Fusion | Hardware Fit | Best For |
|-------|--------|-----------------|---------------|--------|--------------|----------|
| **TinyVLA-1** | ~8M | MobileNetV3-Small | TinyBERT-4L | Concat + MLP | ✅ Both | Simple commands + vision |
| **TinyVLA-2** | ~12M | EfficientNet-B0 | TinyBERT-4L | Cross-Attention | ✅ Both | Better vision-language fusion |
| **TinyVLA-3** | ~18M | MobileNetV3-Large | MobileBERT | FiLM | ❌ Hailo / ✅ Intel | Higher capacity |

---

## **🦀 Rust Tools & Libraries**

### **🔧 Core ML Inference**
| Library | Description | Pros | Cons | Best For |
|---------|-------------|------|------|----------|
| **[tract-onnx](https://github.com/sonos/tract)** | ONNX runtime in pure Rust | Pure Rust, no dependencies, supports Hailo/Intel | Slower than native SDKs | General inference |
| **[tch-rs](https://github.com/LaurentMazare/tch-rs)** | PyTorch bindings for Rust | Access to PyTorch ecosystem | Requires PyTorch C++ backend | Prototyping, CPU/GPU |
| **[burn](https://github.com/burn-rs/burn)** | Pure Rust ML framework | Native Rust, GPU support | Young ecosystem | Training & inference in Rust |
| **[openvino-rs](https://crates.io/crates/openvino)** | OpenVINO bindings | Intel NPU optimized | Intel-only | Intel 358H deployment |
| **[candle](https://github.com/huggingface/candle)** | HF's Rust ML library | HuggingFace compatible | Early stage | HuggingFace model support |

### **📡 Inter-Agent Communication**
| Library | Description | Pros | Cons | Best For |
|---------|-------------|------|------|----------|
| **[tokio](https://tokio.rs/)** | Async runtime | Industry standard, mature | Steep learning curve | Agent orchestration |
| **[zmq-rs](https://crates.io/crates/zmq)** | ZeroMQ bindings | Mature, multi-language | Add overhead | Cross-platform agents |
| **[nanomsg-rs](https://crates.io/crates/nanomsg)** | Nanomsg bindings | Lightweight, simple | Less common | Lightweight IPC |
| **[quinn](https://github.com/quinn-rs/quinn)** | QUIC protocol | Modern, fast | Newer, less tested | Low-latency streaming |
| **[crossbeam](https://github.com/crossbeam-rs/crossbeam)** | Parallelism library | Thread-safe, efficient | Not for distributed | Single-process agents |

### **💾 Data & Serialization**
| Library | Description | Pros | Cons | Best For |
|---------|-------------|------|------|----------|
| **[serde](https://serde.rs/)** | Serialization framework | Ubiquitous, fast | None | All data serialization |
| **[bincode](https://github.com/servo/bincode)** | Binary serialization | Fast, compact | No schema evolution | Internal agent data |
| **[protobuf](https://crates.io/crates/protobuf)** | Protocol Buffers | Schema evolution, multi-language | Verbose | Cross-language agents |
| **[flatbuffers](https://crates.io/crates/flatbuffers)** | Zero-copy serialization | Fast, zero-copy | Complex API | High-performance streaming |
| **[ndarray](https://github.com/rust-ndarray/ndarray)** | N-dimensional arrays | BLAS support, efficient | Learning curve | Tensor operations |

### **🎛️ Robotics & Control**
| Library | Description | Pros | Cons | Best For |
|---------|-------------|------|------|----------|
| **[real-time-rs](https://crates.io/crates/real-time)** | Real-time scheduling | Precise timing | Platform-specific | Control loops |
| **[ncollide](https://ncollide.org/)** | 2D/3D collision detection | Pure Rust, accurate | Not for soft-body | Collision checking |
| **[nalgebra](https://nalgebra.org/)** | Linear algebra | Comprehensive, efficient | Learning curve | IK, kinematics |
| **[kiss3d](https://github.com/sebcrozet/kiss3d)** | 3D visualization | Simple, pure Rust | Limited features | Debug visualization |

### **📦 Model Deployment**
| Library | Description | Pros | Cons | Best For |
|---------|-------------|------|------|----------|
| **[onnxruntime-rs](https://crates.io/crates/onnxruntime)** | ONNX Runtime bindings | Microsoft-backed, optimized | Heavy dependency | General ONNX inference |
| **[tract-hailo](https://github.com/sonos/tract)** | Tract with Hailo backend | Hailo-optimized | Experimental | Hailo-10H deployment |
| **[intel-openvino](https://crates.io/crates/openvino)** | OpenVINO Rust bindings | Intel NPU optimized | Intel-only | Intel 358H |
| **[model-zoo](https://github.com/rust-ml/model-zoo)** | Pre-trained Rust models | Easy to use | Limited selection | Quick prototyping |

---

## **🎯 Model-Task Fit Matrix**

*Which tiny model to use for which task?*

### **Vision Tasks**
| Task | Best Model | Why | Alternatives |
|------|-----------|-----|--------------|
| **Object Detection (Pole, Obstacles)** | Yolo-Nano | Fast, real-time | MobileNetV3 + SSD |
| **Scene Classification** | MobileNetV3-Small | Lightweight, accurate | EfficientNet-B0 |
| **Depth Estimation** | Depth Anything (Tiny) | Specialized | MobileNetV3 + head |
| **Pose Estimation (Robot)** | MobileNetV3 + Kinematic Head | Custom output | EfficientNet-B0 |
| **Feature Extraction** | EfficientNet-B0 | Best accuracy/size | MobileNetV3-Large |

### **Language Tasks**
| Task | Best Model | Why | Alternatives |
|------|-----------|-----|--------------|
| **Command Parsing** | TinyBERT | Lightweight NLU | ALBERT-Tiny |
| **Intent Classification** | TinyBERT | Good for categorizing commands | DistilBERT (if fits) |
| **Structured Output (JSON)** | BART-Tiny | Text-to-structured | TinyBERT + head |
| **Reasoning (Feasibility)** | MobileBERT | Better reasoning | TinyBERT + prompt engineering |

### **Control Tasks**
| Task | Best Model | Why | Alternatives |
|------|-----------|-----|--------------|
| **Joint Position Control** | 2-layer MLP | Simple, fast | 3-layer MLP |
| **Dynamic Movement (Walking)** | 3-layer MLP (RL-trained) | Handles non-linearity | GRU for temporal |
| **Vision-Guided Reaching** | Small CNN + MLP | Processes visual input | Transformer (if fits) |
| **Sequential Tasks** | GRU / LSTM | Memory of past states | Transformer (heavier) |
| **High-Precision Control** | Tiny Transformer | Attention over state history | MLP with wide layers |

### **Fused VLA Tasks**
| Task | Best Model | Why | Alternatives |
|------|-----------|-----|--------------|
| **Simple Commands + Vision** | TinyVLA-1 (8M) | Balanced size/capability | Modular approach |
| **Complex Commands** | TinyVLA-2 (12M) | Better fusion | Modular VLA |
| **Maximum Capacity** | TinyVLA-3 (18M) | Highest accuracy | Only on Intel 358H |

---

## **🏗️ Recommended Agent Architectures**

### **Option 1: 3-Agent Minimal (Best for Edge)**
```
┌─────────────────────────────────────────────────────────────────┐
│                      LANGUAGE AGENT (Intel 358H)                    │
│  Model: TinyBERT (14M)                                               │
│  Input: User Text + Scene Description                              │
│  Output: {skill: "climb", target: "pole", params: {...}}          │
└──────────────────────┬────────────────────────────────────────────┘
                       │ (Shared Memory / ZMQ)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       VISION AGENT (Intel 358H)                    │
│  Model: MobileNetV3-Small (2.5M)                                    │
│  Input: RGB/Depth Camera                                           │
│  Output: {objects: [{name: "pole", pos: [x,y,z], affordances: [...]}]} │
└──────────────────────┬────────────────────────────────────────────┘
                       │ (Shared Memory)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTROL AGENT (Hailo-10H)                     │
│  Model: 3-layer MLP (0.5M) or Small CNN+MLP (5M)                  │
│  Input: Goal + Sensor Data                                          │
│  Output: Joint Torques / Positions                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Latency:** ~30-50ms total (10ms vision + 15ms language + 5ms control)
**Hardware:** Intel 358H (2 agents) + Hailo-10H (1 agent)

---

### **Option 2: 4-Agent with Planning (More Capable)**
```
┌─────────────────────────────────────────────────────────────────┐
│                      LANGUAGE AGENT (Intel 358H)                    │
│  Model: MobileBERT (25M)                                            │
│  Input: User Text + Scene Description                              │
│  Output: {intent: "climb", constraints: ["safety", "speed"]}      │
└──────────────────────┬────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VISION AGENT (Intel 358H)                    │
│  Model: MobileNetV3-Large (5.4M)                                   │
│  Input: RGB/Depth + Previous Frames                               │
│  Output: {scene_graph: {pole: {pos, size, climbable: true}, ...}}  │
└──────────────────────┬────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PLANNING AGENT (Intel 358H)                   │
│  Model: TinyBERT (14M) or Pure Logic                              │
│  Input: Intent + Scene Graph                                       │
│  Output: [Step1, Step2, Step3] (Action Sequence)                   │
└──────────────────────┬────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTROL AGENT (Hailo-10H)                     │
│  Model: Small CNN+MLP (5M) or GRU (3M)                             │
│  Input: Current Step + Sensor Data                                 │
│  Output: Joint Torques                                              │
└─────────────────────────────────────────────────────────────────┘
```

**Latency:** ~50-80ms total
**Hardware:** Intel 358H (3 agents) + Hailo-10H (1 agent)
**Use Case:** Complex multi-step tasks

---

### **Option 3: 2-Agent Fused (Simpler, Less Flexible)**
```
┌─────────────────────────────────────────────────────────────────┐
│                    VISION-LANGUAGE AGENT (Intel 358H)                │
│  Model: TinyVLA-1 (8M) or TinyVLA-2 (12M)                           │
│  Input: User Text + RGB Camera                                      │
│  Output: Direct Joint Actions                                        │
└─────────────────────────────────────────────────────────────────┘
                       │ (Shared Memory)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTROL AGENT (Hailo-10H)                     │
│  Model: 2-layer MLP (0.1M) - Safety Filter & Smoothing             │
│  Input: Raw Actions + Sensor Data                                   │
│  Output: Safe, Smooth Joint Torques                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Latency:** ~20-30ms total
**Hardware:** Intel 358H (1 agent) + Hailo-10H (1 agent)
**Use Case:** Simple tasks, low latency required

---

---

## **🔄 Development Roadmap**

### **Phase 1: Prototyping (1-2 Weeks)**
- **Goal:** Basic text-controlled robot in MuJoCo.
- **Approach:** Rule-Based + IK.
- **Hardware:** CPU (laptop/desktop).
- **Deliverables:**
  - Robot responds to 5-10 text commands.
  - Basic IK for reaching targets.

### **Phase 2: Learning (2-4 Weeks)**
- **Goal:** Adaptive movement (walking, climbing).
- **Approach:** RL (Proprioception or Vision + Proprioception).
- **Hardware:** GPU (training), CPU/Intel (inference).
- **Deliverables:**
  - RL policy for walking/climbing.
  - Vision-based target detection.

### **Phase 3: Language Integration (2-3 Weeks)**
- **Goal:** Natural language commands.
- **Approach:** Modular VLA (SayCan).
- **Hardware:** Intel 358H.
- **Deliverables:**
  - LLM for command parsing.
  - Integration with RL/IL policies.

### **Phase 4: Edge Deployment (2-4 Weeks)**
- **Goal:** Real-time control on edge hardware.
- **Approach:** Multi-Agent or End-to-End VLA.
- **Hardware:** Hailo-10H + Intel 358H.
- **Deliverables:**
  - Quantized models.
  - Hardware-optimized inference.
  - Latency < 20ms per step.

### **Phase 5: Scaling (Ongoing)**
- **Goal:** Multi-robot systems, complex tasks.
- **Approach:** Multi-Agent + VLA.
- **Hardware:** Distributed (multiple Hailo/Intel chips).
- **Deliverables:**
  - Orchestration framework.
  - Shared memory/communication protocols.

---

---

## **🔗 Key Resources**

### **Simulation & Control**
- [MuJoCo](https://mujoco.readthedocs.io/) – Physics engine for robotics.
- [PyBullet](https://pybullet.org/) – Alternative to MuJoCo.
- [Isaac Gym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) – RL training in MuJoCo.

### **Reinforcement Learning**
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) – RL algorithms for PyTorch.
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html) – Scalable RL library.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) – NVIDIA’s robotics framework.

### **Vision-Language Models**
- [RT-2 (Google)](https://arxiv.org/abs/2307.15818) – End-to-End VLA model.
- [SayCan (Google)](https://arxiv.org/abs/2204.01691) – Modular VLA approach.
- [PaLM-E (Google)](https://arxiv.org/abs/2303.03378) – Embodied multimodal language model.
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) – Diffusion models for robotics.

### **Edge Deployment**
- [Hailo SDK](https://hailo.ai/developers/) – Tools for Hailo-10H.
- [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) – Intel’s inference optimization.
- [ONNX Runtime](https://onnxruntime.ai/) – Cross-platform inference.
- [TensorRT](https://developer.nvidia.com/tensorrt) – NVIDIA’s inference accelerator.

### **Datasets for VLA**
- [RT-1 Dataset](https://arxiv.org/abs/2212.06817) – Robot trajectories with text.
- [Language-Table](https://language-table.github.io/) – Language-conditioned tabletop manipulation.
- [RLBench](https://rlbench.com/) – 1000+ robotic tasks.
