# IRWOZ - Industrial Robots Domain Wizard-of-Oz dataset

## Introduction
Industrial Robots Domain Wizard-of-Oz dataset (IRWOZ), a fully-labeled dialogue dataset of human-human conversations spanning 
over four domains (Product Assembly, Transportation, Position, Relocation). At a size of xxx dialogues, it aims to provide 
simulated dialogues between shop floor worker and industrial robots to support language-assisted Human Robot Interaction (HRI) in 
industrial setup. To the best of our knowledge, IRWOZ is the first annotated task-oriented corpora for manufacturing domain.

### Data Structure
To maintain a high scalability, IRWOZ has a similar data structure of the most popular 
Multi-Domain Wizard-of-Oz dataset ([MultiWOZ](https://github.com/budzianowski/multiwoz)). 
Each dialogue consists of a domain, multiple user&system utterances and belief state as well as system act. 

The belief state have two sections: DB_request and T_inform. DB_request refers to slots that need to be
used for query the database. T_inform includes slots which relate to the task. Each of them includes 
required (req) and optional (opt) sections. "req" contains all the slots must be obtained during the
dialogue while the slots in "opt" are the optional. The system act contains all the DB search results and status of the required slots. 

### Real Time Robotics database
IRDB.db is a [sqlite3](https://www.sqlite.org/index.html) database. It includes five tables, area_location, employee, product, object and position, which supports 
grounding system actions. 