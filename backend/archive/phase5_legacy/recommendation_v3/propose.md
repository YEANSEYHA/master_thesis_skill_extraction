# Proposed Solution

Do not retrieve postings directly.

Retrieve Career Domains first.

## Step 1 — Build Career Profiles Offline

From your 142k IT jobs.

Aggregate them into stable career domains.

Example:

```
AI Engineer
Data Scientist
ML Engineer
↓
Artificial Intelligence

Backend Developer
Software Engineer
Full Stack Developer
↓
Software Engineering

Security Analyst
SOC Analyst
Penetration Tester
↓
Cybersecurity
```

Final:

```
AI
Software Engineering
Cybersecurity
Data Science
Cloud & DevOps
Networking
Business IT
```

## Step 2 — Build Domain Embeddings

Each domain gets a profile.

Example:

```
AI
machine learning
deep learning
tensorflow
pytorch
computer vision
nlp

Software Engineering
java
oop
software design
testing
git
database

Cybersecurity
network security
forensics
penetration testing
incident response
```

Store:

```json
{
  "AI": embedding,
  "Software Engineering": embedding,
  "Cybersecurity": embedding
}
```

## Step 3 — Route Course to Domain(s)

Course:

```
Introduction to AI
```

Extracted skills:

```
machine learning
decision tree
k-means
neural network
```

Compare to domain profiles:

```
AI                     0.91
Data Science           0.83
Software Engineering   0.42
Cybersecurity          0.11
```

Select:

```
AI
Data Science
```

## Step 4 — Retrieve Postings

Only now.

Take:

```
AI
Data Science
```

and retrieve all postings assigned to those domains.

Example:

```
AI Engineer
Machine Learning Engineer
Data Scientist
```

## Step 5 — Build Market Inventory

Aggregate skills from selected postings.

```
Python
TensorFlow
PyTorch
MLOps
Model Deployment
```

Now this inventory represents the labor market.

## Why this solves the problem

Instead of:

```
Course
↓
Job Postings
```

you do:

```
Course
↓
Career Domain
↓
Job Postings
```

The intermediate layer acts as a filter.

## If I were defending this

I would state:

A course does not directly correspond to a set of job postings. It corresponds to one or more career domains. Therefore, labor-market analysis should be performed against career-domain-specific postings rather than the entire job corpus.

That sentence alone is probably the strongest conceptual contribution in your current thesis because it explains why AMC-style whole-market coverage becomes noisy and justifies your new pipeline.

The remaining challenge is not the coverage score. It's how to build those career domains in a defensible way. My recommendation is:

```
ESCO Occupation
↓
Career Domain
↓
Job Posting
```

because that gives you an academic foundation instead of inventing categories yourself.
