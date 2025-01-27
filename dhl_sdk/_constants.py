"""
This module defines all the constants used across the project. Constants are 
defined as variables with uppercase names to signify their immutability and 
importance. This file serves as a single source of truth for static values 
to improve code readability, maintainability, and consistency.
"""

PRODUCTS_URL = "api/db/v2/products"
RECIPES_URL = "api/db/v2/recipes"
FILES_URL = "api/db/v2/files"
EXPERIMENTS_URL = "api/db/v2/experiments"
VARIABLES_URL = "api/db/v2/variables"
GROUPS_URL = "api/db/v2/groups"
PROJECTS_URL = "api/db/v2/projects"
DATASETS_URL = "api/db/v2/datasets"
MODELS_URL = "api/db/v2/pipelineJobs"
TEMPLATES_URL = "api/db/v2/pipelineJobTemplates"
PREDICT_URL = "api/pipeline/v1/pipeline"


PROCESS_UNIT_MAP = {
    "cultivation": "04a324da-13a5-470b-94a1-bda6ac87bb86",
    "spectroscopy": "373c173a-1f23-4e56-874e-90ca4702ec0d",
}
