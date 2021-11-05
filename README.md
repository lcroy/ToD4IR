# ToD4IR - Task-Oriented Dialogue System for Industrial Robots

## Introduction

This repository showecase building task-oriented dialogue system, ToD4IR, with the focus on industrial robots,
e.g., [Mobile Industrial Robot](https://www.mobile-industrial-robots.com/en/), 
[Frank Emika](https://www.franka.de/). To enhance the user experience and improve the user engagement, human-to-human
conversation strategies are introduced to generate near human response to provide a more natural and flexible conversation
environment.

## System Architecture
The proposed ToD4IR follows End-to-End patten. It consists of cognitive speech service, human-robot dialogue service and 
robot control service. The following figure illustrates a high-level system architecture of ToD4IR.
![system architecture](https://github.com/lcroy/ToD4IR/tree/main/doc/overall_architecture.png)
Two backbone models, [GPT-2](https://huggingface.co/gpt2) and [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-2.7B), are used. Those
models are trained on large amounts of open Web text and learned how to complete a sentence in a given context.
