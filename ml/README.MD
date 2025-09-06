Struct project:
```
├── data
│   ├── resume_1_buisness_anal.rtf
│   ├── resume_1_IT.docx
│   ├── resume_2_buisness_anal.rtf
│   ├── resume_2_IT.docx
│   ├── vacancyBuisness.docx
│   └── vacancyIT.docx
├── modules
│   ├── extract_data_from_docx.py
│   └── __init__.py
├── README.MD
└── screenCV.py
```

## Example use

```python
from screenCV import InitialScreening

ll = InitialScreening(

    PathToVacancy = "data/vacancyBuisness.docx",
    PathToCV = "data/resume_2_buisness_anal.rtf",
    ApiUrl = "http://localhost:1234/v1", 
    ApiKey = "lm-studio"

    ).check_cv()
```
### Output: (float number)
```bash
85.0
```
