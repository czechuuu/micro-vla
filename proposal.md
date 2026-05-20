# Proposal 2: Micro-VLA: Scaling In-Context Robotic Manipulation via Observation-Only Videos

## Overview & Motivation 
Vision-Language-Action (VLA) models, such as RT-2 and OpenVLA, have recently demonstrated incredible zero-shot generalization in robotics. However, these models face a severe data bottleneck: while we have massive repositories of video data showing humans or robots completing tasks, we have very little action-annotated data containing the exact motor torques and joint commands required to replicate those movements.  

Standard supervised approaches fail when actions are missing. If we can design architectures that learn world dynamics and physics directly from "observation-only" videos, we can vastly expand the training sets for robotic foundation models. This project explores whether pre-training on a large corpus of action-free manipulation videos can improve a small-scale VLA's ability to perform In-Context Learning (ICL) from just one or two fully annotated demonstrations.

## Project Objective 
The primary goal of this project is to build a "Micro-VLA" agent that improves its sample efficiency and few-shot generalization by learning from unannotated video trajectories. The student will implement a system that uses an Inverse Dynamics Model (or masked token prediction) to ingest observation-only sequences from a simulated robotics environment, combining this with a small set of action-annotated data to solve continuous control manipulation tasks.

## Expected Deliverables (Pass Criteria)
To successfully pass this project, students are expected to complete the following concrete tasks:

- **Simulated Benchmark Setup**: Set up a lightweight, continuous-control robotic manipulation environment (e.g., Robomimic or Meta-World). Generate a synthetic dataset: a small fraction containing full state-action-reward data, and a large fraction stripped of actions to simulate "video-only" observations.

- **Architecture Implementation**: Design a small Transformer-based policy (e.g., a miniaturized Decision Transformer). Implement an auxiliary objective, such as an Inverse Dynamics module, that forces the network to predict the missing actions between two consecutive video frames, allowing it to learn from the observation-only dataset.

- **Benchmarking & Evaluation**: Train two models: a baseline Micro-VLA trained strictly on the small, fully-annotated dataset, and your proposed Micro-VLA trained on the mixed dataset.

- **Comparative Analysis**: Deliver a final report evaluating the models on their In-Context Learning capabilities. Specifically, prompt the frozen models with a single successful demonstration of a new, unseen task variation (e.g., picking up a differently colored object) and measure which model exhibits better zero-shot or few-shot transfer.

## Advanced Extensions (For students who want to do more)
If you complete the base requirements quickly, you can explore the following advanced directions:

- **Cross-Embodiment Generalization**: Collect your observation-only video data using one type of robot arm (e.g., a Sawyer arm), but evaluate the In-Context Learning agent on a completely different kinematic structure (e.g., a Franka Emika Panda arm). Investigate whether learning general physics from the first robot transfers to the second.

- **Action Chunking for Error Mitigation**: Inverse dynamics models are prone to compounding errors. Implement an "Action Chunking" mechanism (similar to the ACT architecture) where the Transformer predicts a sequence of future actions rather than just the next step. Evaluate if this temporal smoothing improves the stability of the rollout.

- **Learning from Sub-optimal Videos**: The real world is full of mistakes. Introduce a large percentage of failed trajectories into your observation-only dataset. Modify your architecture to condition on a "success/failure" token or return-to-go. Can the model learn useful dynamics from watching failures without copying the bad behavior?

- **Test-Time Adaptation**: While In-Context Learning is weight-free, explore whether taking the prompt demonstration and performing just 5–10 gradient update steps (online fine-tuning) on the frozen model's final layers dramatically boosts the success rate compared to pure ICL.
