from typing import List
from pydantic import BaseModel, Field


class EmployeeData(BaseModel):
    """Employee data for prediction"""

    employee_id: int = Field(description="Unique identifier for the employee")
    age: int = Field(description="Age of the employee")
    gender: str = Field(description="Gender of the employee (Male/Female/Other)")
    marital_status: str = Field(
        description="Marital status (Single, Married, Divorced, Widowed)"
    )
    salary: float = Field(description="Annual salary in USD")
    employment_type: str = Field(
        description="Employment type (Full-time, Part-time, Contract)"
    )
    region: str = Field(description="Region (Northeast, South, West, Midwest)")
    has_dependents: str = Field(description="Whether employee has dependents (Yes/No)")
    tenure_years: float = Field(description="Years of employment")


class BatchEmployeeData(BaseModel):
    """Batch of employee data for prediction"""

    employees: List[EmployeeData]


class PredictionResponse(BaseModel):
    """Prediction response for a single employee"""

    prediction: int
    probability: float
    enrolled: str
    request_id: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response for multiple employees"""

    predictions: List[PredictionResponse]
    request_id: str
