"""
ISCO-08 Domain Mapper using JobBERT embeddings.

Maps raw job_category strings to ISCO-08 major groups (Level 1)
via cosine similarity against ISCO-08 sub-major group names (Level 2).

Reference: International Standard Classification of Occupations (ILO, 2008)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── ISCO-08 Level 2: 43 sub-major groups ──
# Source: ILO ISCO-08 Structure
# Official names stored separately; expanded descriptions used for embedding
ISCO_SUB_MAJOR_GROUPS = {
    # Major Group 1: Managers
    "11": "Chief Executives, Senior Officials and Legislators",
    "12": "Administrative and Commercial Managers",
    "13": "Production and Specialised Services Managers",
    "14": "Hospitality, Retail and Other Services Managers",
    # Major Group 2: Professionals
    "21": "Science and Engineering Professionals",
    "22": "Health Professionals",
    "23": "Teaching Professionals",
    "24": "Business and Administration Professionals",
    "25": "Information and Communications Technology Professionals",
    "26": "Legal, Social and Cultural Professionals",
    # Major Group 3: Technicians and Associate Professionals
    "31": "Science and Engineering Associate Professionals",
    "32": "Health Associate Professionals",
    "33": "Business and Administration Associate Professionals",
    "34": "Legal, Social, Cultural and Related Associate Professionals",
    "35": "Information and Communications Technology Technicians",
    # Major Group 4: Clerical Support Workers
    "41": "General and Keyboard Clerks",
    "42": "Customer Services Clerks",
    "43": "Numerical and Material Recording Clerks",
    "44": "Other Clerical Support Workers",
    # Major Group 5: Service and Sales Workers
    "51": "Personal Service Workers",
    "52": "Sales Workers",
    "53": "Personal Care Workers",
    "54": "Protective Services Workers",
    # Major Group 6: Skilled Agricultural, Forestry and Fishery Workers
    "61": "Market-oriented Skilled Agricultural Workers",
    "62": "Market-oriented Skilled Forestry, Fishery and Hunting Workers",
    "63": "Subsistence Farmers, Fishers, Hunters and Gatherers",
    # Major Group 7: Craft and Related Trades Workers
    "71": "Building and Related Trades Workers, Excluding Electricians",
    "72": "Metal, Machinery and Related Trades Workers",
    "73": "Handicraft and Printing Workers",
    "74": "Electrical and Electronic Trades Workers",
    "75": "Food Processing, Woodworking, Garment and Other Craft and Related Trades Workers",
    # Major Group 8: Plant and Machine Operators, and Assemblers
    "81": "Stationary Plant and Machine Operators",
    "82": "Assemblers",
    "83": "Drivers and Mobile Plant Operators",
    # Major Group 9: Elementary Occupations
    "91": "Cleaners and Helpers",
    "92": "Agricultural, Forestry and Fishery Labourers",
    "93": "Labourers in Mining, Construction, Manufacturing and Transport",
    "94": "Food Preparation Assistants",
    "95": "Street and Related Sales and Service Workers",
    "96": "Refuse Workers and Other Elementary Workers",
    # Major Group 0: Armed Forces Occupations
    "01": "Commissioned Armed Forces Officers",
    "02": "Non-commissioned Armed Forces Officers",
    "03": "Armed Forces Occupations, Other Ranks",
}

# Expanded descriptions for better embedding matching.
# Short category names like "HR", "IT", "Agriculture" need richer context
# to match correctly via cosine similarity.
ISCO_EXPANDED = {
    "11": "Chief Executives, Senior Officials, Legislators, CEO, Director, Executive Management",
    "12": "Administrative and Commercial Managers, Marketing Manager, Sales Manager, Advertising, PR, Public Relations",
    "13": "Production and Specialised Services Managers, IT Manager, Construction Manager, Logistics Manager, Supply Chain",
    "14": "Hospitality, Retail, Hotel, Restaurant, Tourism, Travel, Food and Beverage, Service Managers",
    "21": "Science and Engineering Professionals, Mechanical Engineer, Civil Engineer, Electrical Engineer, Chemical Engineer, Architecture, Construction, Design",
    "22": "Health Professionals, Doctor, Nurse, Pharmacy, Medical, Hospital, Clinical, Psychology, Healthcare",
    "23": "Teaching Professionals, Education, Training, Academic, School, University, Lecturer, Teacher, Educator",
    "24": "Business and Administration Professionals, Finance, Accounting, Audit, Banking, Insurance, HR, Human Resources, Recruitment, Talent, Business Administration, Consulting, Advisory",
    "25": "Information and Communications Technology Professionals, IT, Software, Programming, Web Development, Data Science, Computer Science, Cybersecurity, Network, Digital, ICT, Telecommunications",
    "26": "Legal, Social and Cultural Professionals, Law, Compliance, Social Work, Community Development, NGO, Translation, Media, Journalism, Library, Arts, Graphic Design, Creative, Interior Design, UX UI Design",
    "31": "Science and Engineering Associate Professionals, Lab Technician, Quality Control, Technical Support, Engineering Technician",
    "32": "Health Associate Professionals, Nursing Associate, Dental Hygienist, Pharmacy Technician, Medical Records",
    "33": "Business and Administration Associate Professionals, Accounting Clerk, Insurance Agent, Real Estate Agent, Operations, Admin, Supervisory",
    "34": "Legal, Social, Cultural Associate Professionals, Sports, Fitness, Entertainment, Religious, Community Worker",
    "35": "Information and Communications Technology Technicians, IT Support, Network Technician, Web Technician, Help Desk",
    "41": "General and Keyboard Clerks, Secretary, Data Entry, Office Clerk, Administrative Assistant",
    "42": "Customer Services Clerks, Customer Service, Customer Support, Call Centre, Receptionist, Front Desk",
    "43": "Numerical and Material Recording Clerks, Accounting Clerk, Stock Clerk, Inventory, Logistics Clerk, Procurement Clerk, Warehouse",
    "44": "Other Clerical Support Workers, Library Clerk, Mail Carrier, Filing Clerk",
    "51": "Personal Service Workers, Hairdresser, Beauty, Cosmetic, Skincare, Cook, Chef, Funeral, Astrologer",
    "52": "Sales Workers, Sales, Retail, Wholesale, Shop, Marketing, Business Development, Trading, Import Export, FMCG",
    "53": "Personal Care Workers, Childcare, Elderly Care, Home Care, Domestic Worker",
    "54": "Protective Services Workers, Security, Fire, Police, Guard, Safety",
    "61": "Market-oriented Skilled Agricultural Workers, Agriculture, Farming, Agronomy, Crop, Livestock, Horticulture, Rural Development, Forestry, Environment, Natural Resources",
    "62": "Market-oriented Skilled Forestry, Fishery and Hunting Workers, Forestry, Fishing, Aquaculture, Environment, Natural Resources",
    "63": "Subsistence Farmers, Fishers, Hunters and Gatherers",
    "71": "Building and Related Trades Workers, Construction, Architecture, Building, Masonry, Carpentry, Plumbing, Painting",
    "72": "Metal, Machinery and Related Trades Workers, Welding, Blacksmith, Toolmaking, Mechanical, Automotive, Heavy Equipment, Vehicle Repair, Motor Mechanic",
    "73": "Handicraft and Printing Workers, Printing, Bookbinding, Handicraft, Jewelry, Pottery",
    "74": "Electrical and Electronic Trades Workers, Electrician, Electronics, Electrical Installation, Telecommunications Installer",
    "75": "Food Processing, Woodworking, Garment Workers, Baking, Butcher, Tailor, Furniture, Textile, Fashion",
    "81": "Stationary Plant and Machine Operators, Manufacturing, Factory, Production, Assembly Line, Machine Operator",
    "82": "Assemblers, Electronics Assembly, Mechanical Assembly, Product Assembly",
    "83": "Drivers and Mobile Plant Operators, Driver, Transport, Delivery, Crane Operator, Forklift, Transportation, Logistics, Shipping",
    "91": "Cleaners and Helpers, Housekeeping, Cleaning, Janitorial, Office Cleaner",
    "92": "Agricultural, Forestry and Fishery Labourers, Farm Labourer, Forestry Labourer",
    "93": "Labourers in Mining, Construction, Manufacturing and Transport, Mining Labourer, Construction Labourer",
    "94": "Food Preparation Assistants, Kitchen Helper, Food Service, Fast Food Worker",
    "95": "Street and Related Sales and Service Workers, Street Vendor, Car Washer, Window Cleaner",
    "96": "Refuse Workers and Other Elementary Workers, Garbage Collector, Sweeper",
    "01": "Commissioned Armed Forces Officers, Military Officer, Army Officer",
    "02": "Non-commissioned Armed Forces Officers, Military Sergeant, Army NCO",
    "03": "Armed Forces Occupations Other Ranks, Military Soldier, Armed Forces",
}

# ISCO-08 Level 1: 10 major groups
ISCO_MAJOR_GROUPS = {
    "0": "Armed Forces Occupations",
    "1": "Managers",
    "2": "Professionals",
    "3": "Technicians and Associate Professionals",
    "4": "Clerical Support Workers",
    "5": "Service and Sales Workers",
    "6": "Skilled Agricultural, Forestry and Fishery Workers",
    "7": "Craft and Related Trades Workers",
    "8": "Plant and Machine Operators and Assemblers",
    "9": "Elementary Occupations",
}

# Short display labels for ISCO Level 2 (used in UI domain tabs)
ISCO_SHORT_LABELS = {
    "11": "Executives",
    "12": "Commercial Managers",
    "13": "Production Managers",
    "14": "Hospitality Managers",
    "21": "Engineering",
    "22": "Health",
    "23": "Education",
    "24": "Business & Admin",
    "25": "ICT",
    "26": "Legal & Social",
    "31": "Engineering Technicians",
    "32": "Health Technicians",
    "33": "Admin Associates",
    "34": "Social & Cultural Associates",
    "35": "ICT Technicians",
    "41": "Office Clerks",
    "42": "Customer Service",
    "43": "Recording Clerks",
    "44": "Clerical Support",
    "51": "Personal Service",
    "52": "Sales",
    "53": "Care Workers",
    "54": "Security",
    "61": "Agriculture",
    "62": "Forestry & Fishery",
    "63": "Subsistence Farming",
    "71": "Construction",
    "72": "Machinery & Metal",
    "73": "Handicraft",
    "74": "Electrical Trades",
    "75": "Food & Garment",
    "81": "Machine Operators",
    "82": "Assemblers",
    "83": "Drivers & Transport",
    "91": "Cleaners",
    "92": "Agricultural Labour",
    "93": "Construction Labour",
    "94": "Food Assistants",
    "95": "Street Workers",
    "96": "Refuse Workers",
    "01": "Military Officers",
    "02": "Military NCOs",
    "03": "Armed Forces",
}

# Minimum similarity threshold — below this, map to "Other"
MIN_SIMILARITY = 0.3

# Manual overrides for categories that JobBERT can't match well
# (non-English text, ambiguous short strings, known mismatches)
MANUAL_OVERRIDES = {
    "unknown": ("24", "Business and Administration Professionals"),
    "other": ("24", "Business and Administration Professionals"),
    "others": ("24", "Business and Administration Professionals"),
    "agriculture": ("61", "Market-oriented Skilled Agricultural Workers"),
    "កសិកម្ម": ("61", "Market-oriented Skilled Agricultural Workers"),  # Khmer: agriculture
    "អប់រំ": ("23", "Teaching Professionals"),  # Khmer: education
    "ផ្នែកអប់រំ/ហ្វឹកហ្វឺន/បង្រៀន": ("23", "Teaching Professionals"),  # Khmer: education/training/teaching
    "ផ្នែកអប់រំ/ហ្វឹកហ្វឺន": ("23", "Teaching Professionals"),  # Khmer: education/training
    "បើកបរ/ដឹកជញ្ចូន": ("83", "Drivers and Mobile Plant Operators"),  # Khmer: driving/transport
    "行政": ("24", "Business and Administration Professionals"),  # Chinese: administration
    "economics": ("24", "Business and Administration Professionals"),
    "design": ("26", "Legal, Social and Cultural Professionals"),
}

# Cache
_model = None
_isco_embeddings = None
_isco_codes = None
_isco_labels = None


def _load_model():
    """Load JobBERT model and pre-compute ISCO embeddings (once)."""
    global _model, _isco_embeddings, _isco_codes, _isco_labels
    if _model is not None:
        return

    _model = SentenceTransformer("jjzha/jobbert-base-cased")
    _isco_codes = list(ISCO_SUB_MAJOR_GROUPS.keys())
    _isco_labels = list(ISCO_SUB_MAJOR_GROUPS.values())
    # Use expanded descriptions for embedding (richer context = better matching)
    expanded = [ISCO_EXPANDED[code] for code in _isco_codes]
    _isco_embeddings = _model.encode(expanded, normalize_embeddings=True)
    # Ensure float32 to avoid matmul overflow on some platforms
    _isco_embeddings = np.array(_isco_embeddings, dtype=np.float32)


def _check_override(category: str):
    """Check if category has a manual override."""
    key = category.lower().strip()
    if key in MANUAL_OVERRIDES:
        sub_code, sub_label = MANUAL_OVERRIDES[key]
        major_code = sub_code[0]
        return {
            "category": category,
            "isco_code": sub_code,
            "isco_label": sub_label,
            "major_code": major_code,
            "major_label": ISCO_MAJOR_GROUPS[major_code],
            "similarity": 1.0,
        }
    return None


def map_category_to_isco(category: str) -> dict:
    """
    Map a single job_category string to an ISCO-08 major group.

    Returns:
        {
            "category": original category,
            "isco_code": 2-digit sub-major code (e.g. "25"),
            "isco_label": sub-major group name,
            "major_code": 1-digit major code (e.g. "2"),
            "major_label": major group name (e.g. "Professionals"),
            "similarity": cosine similarity score,
        }
    """
    # Check manual overrides first
    override = _check_override(category)
    if override:
        return override

    _load_model()

    cat_embedding = np.array(
        _model.encode([category], normalize_embeddings=True), dtype=np.float32
    )
    similarities = cosine_similarity(cat_embedding, _isco_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < MIN_SIMILARITY:
        return {
            "category": category,
            "isco_code": None,
            "isco_label": None,
            "major_code": None,
            "major_label": "Other",
            "similarity": best_score,
        }

    sub_code = _isco_codes[best_idx]
    major_code = sub_code[0]

    return {
        "category": category,
        "isco_code": sub_code,
        "isco_label": _isco_labels[best_idx],
        "major_code": major_code,
        "major_label": ISCO_MAJOR_GROUPS[major_code],
        "similarity": best_score,
    }


def map_categories_batch(categories: list[str]) -> dict[str, dict]:
    """
    Map a list of unique category strings to ISCO-08 in one batch.

    Returns: { category_string: mapping_dict }
    """
    if not categories:
        return {}

    # Separate overridden vs needs-embedding
    result = {}
    to_embed = []
    to_embed_idx = []
    for i, category in enumerate(categories):
        override = _check_override(category)
        if override:
            result[category] = override
        else:
            to_embed.append(category)
            to_embed_idx.append(i)

    if not to_embed:
        return result

    _load_model()

    cat_embeddings = np.array(
        _model.encode(to_embed, normalize_embeddings=True), dtype=np.float32
    )
    similarities = cosine_similarity(cat_embeddings, _isco_embeddings)

    for j, category in enumerate(to_embed):
        best_idx = int(np.argmax(similarities[j]))
        best_score = float(similarities[j][best_idx])

        if best_score < MIN_SIMILARITY:
            result[category] = {
                "category": category,
                "isco_code": None,
                "isco_label": None,
                "major_code": None,
                "major_label": "Other",
                "similarity": best_score,
            }
        else:
            sub_code = _isco_codes[best_idx]
            major_code = sub_code[0]
            result[category] = {
                "category": category,
                "isco_code": sub_code,
                "isco_label": _isco_labels[best_idx],
                "major_code": major_code,
                "major_label": ISCO_MAJOR_GROUPS[major_code],
                "similarity": best_score,
            }

    return result
