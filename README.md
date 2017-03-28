# NTUCB

## Installation
Run ``sh install.sh`` and you may access django admin by [http://localhost:8000/admin](http://localhost:8000/admin) or make queries using ``python3 manage.py shell``

## Template generation
``python3 generate_template.py``
Generated files will be put in ``request_template`` folder.

```
request_template
├── [   90775360]  classroom.txt
├── [   65659689]  instructor.txt
├── [   87333770]  schedule.txt
└── [   79905920]  title.txt
```
~1M sentences for each goal.

## Current DB schema:
![](db_schema.png)
