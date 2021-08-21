# STEToD - Small Talk Enhanced Task-Oriented Dialogue System

## Introduction

This repository showecase building task-oriented dialogue system,Max, with the focus on industrial robots,
e.g., [Mobile Industrial Robot](https://www.mobile-industrial-robots.com/en/), 
[Frank Emika](https://www.franka.de/). To enhance the user experience and improve the user engagement, human-to-human
conversation strategies are introduced to generate near human response to provide a more natural and flexible conversation
environment.

We propose xxx We show xxx

## T2IR
Talk To Industrial Robot (T2IR), a fully-labeled dialogue dataset of human-human conversations spanning 
over xxx domains and topics. At a size of xxx dialogues, it aims to provide simulated dialogues between
shop floor worker and industrial robots to support language-assisted Human Robot Interaction (HRI) in 
industrial setup. To the best of our knowledge, T2IR is the first annotated task-oriented corpora for 
manufacturing domain.

### Data Structure
There are xxx single-domain dialogues and xxx multi-domain dialogues consisting of at least x up to x domains. 

To maintain a high scalability, T2IR is constructed by following data structure of the most popular 
Multi-Domain Wizard-of-Oz dataset ([MultiWOZ](https://github.com/budzianowski/multiwoz)). 
Each dialogue consists of a goal, multiple user and system utterances as well as a belief state. 


The belief state have three sections: semi, record and recorded. Semi refers to slots from a particular 
domain. Record refers to recording operators' activities for a particular task. It includes slots which 
need to be updated to the system database, for example, task name, employee's name, timestamp. It helps
to track the manufacturing tasks. Recorded is a sub-list of record dictionary with information about 
the recorded entity (once the recording has been made).