// ── Seeded random for reproducibility ──
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// ── Job roles by domain ──
const ROLES = {
  engineering: [
    "Software Engineer", "Full Stack Developer", "Frontend Developer", "Backend Developer",
    "DevOps Engineer", "Site Reliability Engineer", "Platform Engineer", "Cloud Engineer",
    "Infrastructure Engineer", "Mobile Developer", "iOS Developer", "Android Developer",
    "Embedded Systems Engineer", "Firmware Engineer", "Systems Programmer",
    "API Developer", "Microservices Engineer", "Release Engineer", "Build Engineer",
  ],
  data: [
    "Data Analyst", "Data Scientist", "Machine Learning Engineer", "AI Research Scientist",
    "Data Engineer", "Business Intelligence Analyst", "Analytics Engineer",
    "MLOps Engineer", "NLP Engineer", "Computer Vision Engineer",
    "Quantitative Analyst", "Research Data Scientist", "Statistical Analyst",
  ],
  security: [
    "Cybersecurity Analyst", "Security Engineer", "Penetration Tester",
    "Security Operations Analyst", "Information Security Manager", "SOC Analyst",
    "Application Security Engineer", "Cloud Security Engineer",
  ],
  infra: [
    "Network Administrator", "System Administrator", "Database Administrator",
    "IT Support Specialist", "Help Desk Technician", "Network Engineer",
    "Cloud Architect", "Solutions Architect", "Technical Support Engineer",
  ],
  qa: [
    "QA Engineer", "Test Automation Engineer", "SDET",
    "Performance Test Engineer", "QA Lead", "Manual Tester",
  ],
  product: [
    "Product Manager", "Product Owner", "Technical Product Manager",
    "Product Analyst", "Growth Product Manager",
  ],
  project: [
    "Project Manager", "Scrum Master", "Agile Coach",
    "Program Manager", "Delivery Manager", "PMO Analyst",
  ],
  design: [
    "UX Designer", "UI Designer", "UX/UI Designer", "Graphic Designer",
    "Visual Designer", "Interaction Designer", "Product Designer",
    "Motion Designer", "Brand Designer", "Design System Lead",
  ],
  marketing: [
    "Marketing Manager", "Digital Marketing Specialist", "SEO Specialist",
    "Content Writer", "Content Strategist", "Social Media Manager",
    "PPC Specialist", "Email Marketing Manager", "Growth Hacker",
    "Brand Manager", "Marketing Analyst", "Performance Marketing Manager",
    "Influencer Marketing Coordinator", "Community Manager",
  ],
  sales: [
    "Sales Executive", "Business Development Manager", "Account Manager",
    "Account Executive", "Sales Engineer", "Technical Sales Consultant",
    "Inside Sales Representative", "Regional Sales Manager",
    "Key Account Manager", "Channel Sales Manager", "Pre-Sales Consultant",
  ],
  support: [
    "Customer Service Representative", "Customer Success Manager",
    "Technical Account Manager", "Client Relations Manager",
    "Support Engineer", "Customer Experience Specialist",
  ],
  hr: [
    "HR Manager", "Recruitment Specialist", "Talent Acquisition Lead",
    "HR Business Partner", "People Operations Manager",
    "Compensation & Benefits Analyst", "Learning & Development Specialist",
    "Employee Engagement Manager", "HR Coordinator",
  ],
  finance: [
    "Financial Analyst", "Accountant", "Auditor", "Tax Specialist",
    "Controller", "Treasury Analyst", "Risk Analyst",
    "Investment Analyst", "Compliance Officer", "Payroll Specialist",
    "Bookkeeper", "Credit Analyst", "Budget Analyst",
  ],
  operations: [
    "Operations Manager", "Supply Chain Manager", "Logistics Coordinator",
    "Procurement Specialist", "Warehouse Manager", "Fleet Manager",
    "Inventory Analyst", "Import/Export Coordinator", "Demand Planner",
  ],
  engineering_trad: [
    "Civil Engineer", "Mechanical Engineer", "Electrical Engineer",
    "Chemical Engineer", "Structural Engineer", "Environmental Engineer",
    "Industrial Engineer", "Manufacturing Engineer", "Process Engineer",
    "Biomedical Engineer", "Aerospace Engineer", "Materials Engineer",
  ],
  healthcare: [
    "Nurse", "Pharmacist", "Medical Lab Technician",
    "Radiologist", "Physical Therapist", "Occupational Therapist",
    "Clinical Research Coordinator", "Healthcare Administrator",
    "Dental Hygienist", "Paramedic", "Medical Records Specialist",
  ],
  legal: [
    "Legal Counsel", "Paralegal", "Compliance Analyst",
    "Contract Specialist", "Legal Assistant", "Regulatory Affairs Specialist",
  ],
  education: [
    "Training Specialist", "Instructional Designer", "E-Learning Developer",
    "Academic Coordinator", "Curriculum Developer", "Education Consultant",
  ],
};

const LEVELS = [
  "", "Junior ", "Senior ", "Lead ", "Principal ", "Staff ",
  "Associate ", "Mid-Level ", "Head of ",
];

// ── Skills mapped to domains ──
const DOMAIN_SKILLS = {
  engineering: {
    primary: ["JavaScript", "Python", "Java", "TypeScript", "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask", "Spring Boot", "Ruby on Rails", "PHP", "Laravel", "C#", ".NET", "C++", "Rust", "Go", "Swift", "Kotlin", "Dart", "Flutter", "React Native"],
    tools: ["Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Jenkins", "GitHub Actions", "CircleCI", "Terraform", "Ansible", "Nginx", "Apache", "Webpack", "Vite"],
    db: ["SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "DynamoDB", "Cassandra", "SQLite", "Firebase"],
    concepts: ["REST API", "GraphQL", "microservices", "CI/CD", "agile methodology", "TDD", "design patterns", "system design", "distributed systems", "event-driven architecture"],
  },
  data: {
    primary: ["Python", "R programming", "SQL", "machine learning", "deep learning", "statistics", "data analysis", "data visualization", "data modeling", "ETL pipelines"],
    tools: ["TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "Spark", "Hadoop", "Airflow", "dbt", "Tableau", "Power BI", "Looker", "Jupyter", "MLflow", "Databricks"],
    concepts: ["NLP", "computer vision", "reinforcement learning", "A/B testing", "hypothesis testing", "regression analysis", "clustering", "feature engineering", "model deployment", "big data"],
  },
  security: {
    primary: ["network security", "penetration testing", "vulnerability assessment", "incident response", "SIEM", "firewalls", "encryption", "identity management", "threat modeling", "forensics"],
    tools: ["Splunk", "Wireshark", "Nessus", "Burp Suite", "Metasploit", "Nmap", "OWASP ZAP", "CrowdStrike", "Palo Alto", "Fortinet"],
  },
  infra: {
    primary: ["Linux", "Windows Server", "networking", "TCP/IP", "DNS", "DHCP", "Active Directory", "VMware", "Hyper-V", "load balancing"],
    tools: ["AWS", "Azure", "GCP", "Terraform", "Ansible", "Puppet", "Chef", "Docker", "Kubernetes", "Prometheus", "Grafana", "Nagios", "CloudFormation"],
  },
  qa: {
    primary: ["test automation", "manual testing", "Selenium", "Cypress", "Playwright", "JMeter", "Postman", "API testing", "performance testing", "regression testing"],
    concepts: ["test planning", "bug tracking", "JIRA", "test cases", "CI/CD", "BDD", "TDD", "load testing", "security testing", "mobile testing"],
  },
  product: {
    primary: ["product strategy", "roadmap planning", "user research", "A/B testing", "data analysis", "market analysis", "competitive analysis", "feature prioritization", "OKRs", "KPI tracking"],
    tools: ["JIRA", "Confluence", "Figma", "Amplitude", "Mixpanel", "Google Analytics", "Hotjar", "Miro", "Notion", "Productboard"],
  },
  project: {
    primary: ["project management", "agile methodology", "Scrum", "Kanban", "risk management", "stakeholder management", "budgeting", "resource planning", "change management", "process improvement"],
    tools: ["JIRA", "Confluence", "Microsoft Project", "Asana", "Trello", "Monday.com", "Smartsheet", "Microsoft Office skills"],
  },
  design: {
    primary: ["UI design", "UX research", "wireframing", "prototyping", "responsive design", "design systems", "user testing", "information architecture", "interaction design", "visual design"],
    tools: ["Figma", "Sketch", "Adobe XD", "Adobe Photoshop", "Adobe Illustrator", "InVision", "Principle", "After Effects", "Zeplin", "Framer"],
    concepts: ["typography", "color theory", "brand identity", "accessibility", "mobile-first design", "design thinking", "user personas", "journey mapping"],
  },
  marketing: {
    primary: ["SEO", "SEM", "content strategy", "social media marketing", "email marketing", "PPC advertising", "copywriting", "brand management", "marketing automation", "lead generation"],
    tools: ["Google Analytics", "Google Ads", "Facebook Ads Manager", "HubSpot", "Mailchimp", "Hootsuite", "SEMrush", "Ahrefs", "Canva", "Buffer", "Marketo", "Salesforce Marketing Cloud"],
    concepts: ["conversion optimization", "customer segmentation", "marketing funnel", "content marketing", "influencer marketing", "growth hacking", "campaign management"],
  },
  sales: {
    primary: ["business development", "negotiation", "account management", "sales strategy", "pipeline management", "lead generation", "cold calling", "relationship building", "solution selling", "consultative selling"],
    tools: ["Salesforce", "HubSpot CRM", "LinkedIn Sales Navigator", "Pipedrive", "Zoho CRM", "Microsoft Office skills", "PowerPoint"],
    concepts: ["B2B sales", "B2C sales", "territory management", "quota management", "contract negotiation", "upselling", "cross-selling"],
  },
  support: {
    primary: ["customer service", "technical support", "troubleshooting", "customer onboarding", "customer retention", "escalation management", "SLA management", "ticket management"],
    tools: ["Zendesk", "Intercom", "Freshdesk", "Salesforce Service Cloud", "JIRA Service Management", "LiveChat"],
    concepts: ["customer satisfaction", "NPS", "first response time", "churn reduction", "customer advocacy"],
  },
  hr: {
    primary: ["recruitment", "talent acquisition", "employee relations", "performance management", "compensation & benefits", "onboarding", "training & development", "workforce planning", "HR policy", "diversity & inclusion"],
    tools: ["Workday", "BambooHR", "Greenhouse", "Lever", "LinkedIn Recruiter", "ADP", "SAP SuccessFactors", "Microsoft Office skills"],
    concepts: ["employment law", "organizational development", "succession planning", "employer branding", "people analytics"],
  },
  finance: {
    primary: ["financial analysis", "accounting", "budgeting", "forecasting", "auditing", "tax compliance", "financial reporting", "cost analysis", "cash flow management", "risk assessment"],
    tools: ["Excel", "SAP", "QuickBooks", "Oracle Financials", "Bloomberg Terminal", "Tableau", "Power BI", "Microsoft Office skills"],
    concepts: ["GAAP", "IFRS", "financial modeling", "variance analysis", "internal controls", "due diligence", "valuation", "regulatory compliance"],
  },
  operations: {
    primary: ["supply chain management", "logistics", "inventory management", "procurement", "vendor management", "warehouse operations", "demand forecasting", "quality assurance", "process optimization", "lean management"],
    tools: ["SAP", "Oracle SCM", "Microsoft Office skills", "Excel", "Tableau", "Power BI", "ERP"],
    concepts: ["Six Sigma", "Kaizen", "just-in-time", "cost reduction", "KPI tracking", "SOP development"],
  },
  engineering_trad: {
    primary: ["AutoCAD", "SolidWorks", "MATLAB", "technical drawing", "structural analysis", "project management", "quality control", "safety compliance", "materials testing", "process optimization"],
    tools: ["ANSYS", "Revit", "CATIA", "PLC programming", "SCADA", "Microsoft Office skills", "SAP"],
    concepts: ["building codes", "environmental regulations", "FEA", "thermodynamics", "fluid mechanics", "electrical circuits", "GMP", "ISO standards"],
  },
  healthcare: {
    primary: ["patient care", "clinical procedures", "medical records", "HIPAA compliance", "medication administration", "vital signs monitoring", "infection control", "patient assessment", "emergency response", "health education"],
    tools: ["Epic", "Cerner", "HL7/FHIR", "medical devices", "laboratory equipment", "Microsoft Office skills"],
    concepts: ["evidence-based practice", "patient safety", "interdisciplinary care", "clinical trials", "pharmacology", "anatomy & physiology"],
  },
  legal: {
    primary: ["contract drafting", "legal research", "regulatory compliance", "corporate governance", "intellectual property", "litigation support", "due diligence", "policy development"],
    tools: ["Westlaw", "LexisNexis", "Microsoft Office skills", "document management systems"],
    concepts: ["employment law", "data privacy", "GDPR", "commercial law", "risk assessment"],
  },
  education: {
    primary: ["curriculum development", "instructional design", "e-learning", "learning management systems", "training delivery", "assessment design", "adult learning theory", "educational technology"],
    tools: ["Articulate Storyline", "Adobe Captivate", "Moodle", "Canvas LMS", "Microsoft Office skills", "Google Workspace"],
    concepts: ["learning outcomes", "competency-based education", "blended learning", "gamification", "accessibility"],
  },
};

// ── Soft skills ──
const UNIVERSAL_SOFT = [
  "communication skills", "teamwork", "problem solving", "leadership",
  "time management", "critical thinking", "adaptability", "creativity",
  "attention to detail", "interpersonal skills", "emotional intelligence",
  "presentation skills", "conflict resolution", "decision making",
  "work ethic", "organizational skills", "multitasking", "collaboration",
];

const LANGUAGES = [
  "English language", "Khmer language", "Chinese language",
  "Japanese language", "French language", "Korean language",
  "Thai language", "Vietnamese language", "Spanish language",
];

// ── Domain weights ──
const DOMAIN_WEIGHTS = {
  engineering: 25,
  data: 12,
  security: 5,
  infra: 8,
  qa: 5,
  product: 4,
  project: 5,
  design: 5,
  marketing: 7,
  sales: 8,
  support: 4,
  hr: 3,
  finance: 5,
  operations: 4,
  engineering_trad: 3,
  healthcare: 4,
  legal: 2,
  education: 2,
};

// ── Time-dependent skill probabilities (2022–2026) ──
// Each skill has a base weight + a yearly modifier.
// This creates realistic trends: some skills grow, some decline, some stay stable.
//
// Format: { skill, base, trend }
//   base  = probability in 2022 (0–1 scale, relative)
//   trend = per-year change (+0.04 means +4% probability per year from 2022)
//
// Positive trend = growing skill, negative = declining, ~0 = stable
//
// ── AI boom skills: emerged suddenly after ChatGPT launch (Nov 2022 = offset 0.83) ──
// These skills should be 0 before Nov 2022, then ramp up.
const AI_BOOM_START = 10 / 12; // Nov 2022 in yearOffset units
const AI_BOOM_SKILLS = new Set([
  "prompt engineering", "LLM", "generative AI", "RAG",
  "vector databases", "LangChain",
]);

const SKILL_TRENDS = [
  // ── SURGING (strong growth 2022→2026) ──
  { skill: "TypeScript",          base: 0.08, trend: +0.035 },  // replacing JS
  { skill: "Rust",                base: 0.01, trend: +0.020 },  // systems programming boom
  { skill: "Kubernetes",          base: 0.04, trend: +0.025 },  // container orchestration
  { skill: "Terraform",           base: 0.03, trend: +0.020 },  // IaC adoption
  { skill: "GitHub Actions",      base: 0.02, trend: +0.025 },  // CI/CD shift
  { skill: "PyTorch",             base: 0.03, trend: +0.030 },  // ML framework war winner
  { skill: "dbt",                 base: 0.01, trend: +0.020 },  // data transformation
  { skill: "Databricks",          base: 0.01, trend: +0.015 },  // data platform
  { skill: "Go",                  base: 0.03, trend: +0.015 },  // backend growth
  { skill: "Vite",                base: 0.00, trend: +0.020 },  // replacing Webpack
  { skill: "Playwright",          base: 0.00, trend: +0.018 },  // replacing Selenium
  { skill: "GCP",                 base: 0.03, trend: +0.012 },  // cloud growth
  { skill: "Figma",               base: 0.04, trend: +0.020 },  // design tool dominance
  { skill: "Notion",              base: 0.01, trend: +0.015 },  // productivity
  { skill: "MLflow",              base: 0.01, trend: +0.012 },  // MLOps
  // AI boom skills — boosted trends because effective window starts Nov 2022
  { skill: "prompt engineering",  base: 0.00, trend: +0.050 },  // AI boom post-ChatGPT
  { skill: "LLM",                 base: 0.00, trend: +0.045 },  // large language models
  { skill: "generative AI",       base: 0.00, trend: +0.048 },  // GenAI explosion
  { skill: "RAG",                 base: 0.00, trend: +0.032 },  // retrieval augmented gen
  { skill: "vector databases",    base: 0.00, trend: +0.024 },  // AI infrastructure
  { skill: "LangChain",           base: 0.00, trend: +0.026 },  // AI framework
  { skill: "Next.js",             base: 0.02, trend: +0.018 },  // React meta-framework
  { skill: "Tailwind CSS",        base: 0.02, trend: +0.020 },  // utility CSS
  { skill: "Cypress",             base: 0.02, trend: +0.012 },  // modern testing

  // ── GROWING (moderate growth) ──
  { skill: "Python",              base: 0.20, trend: +0.015 },  // still growing
  { skill: "React",               base: 0.14, trend: +0.010 },  // dominant framework
  { skill: "AWS",                 base: 0.12, trend: +0.010 },  // cloud leader
  { skill: "Docker",              base: 0.10, trend: +0.010 },  // containerization
  { skill: "Node.js",             base: 0.09, trend: +0.008 },  // backend JS
  { skill: "PostgreSQL",          base: 0.06, trend: +0.010 },  // DB of choice
  { skill: "CI/CD",               base: 0.07, trend: +0.008 },  // DevOps practice
  { skill: "GraphQL",             base: 0.03, trend: +0.008 },  // API alternative
  { skill: "Azure",               base: 0.06, trend: +0.008 },  // cloud #2
  { skill: "data analysis",       base: 0.10, trend: +0.008 },  // growing demand
  { skill: "machine learning",    base: 0.06, trend: +0.012 },  // AI growth
  { skill: "deep learning",       base: 0.03, trend: +0.010 },  // AI depth
  { skill: "Power BI",            base: 0.05, trend: +0.008 },  // BI growth
  { skill: "Airflow",             base: 0.02, trend: +0.008 },  // data orchestration
  { skill: "microservices",       base: 0.06, trend: +0.006 },  // architecture
  { skill: "Canva",               base: 0.02, trend: +0.010 },  // design democratization
  { skill: "HubSpot",             base: 0.03, trend: +0.008 },  // marketing platform

  // ── STABLE (minimal change) ──
  { skill: "SQL",                 base: 0.18, trend: +0.002 },  // evergreen
  { skill: "JavaScript",          base: 0.16, trend: -0.002 },  // stable, slight TS migration
  { skill: "Java",                base: 0.10, trend: -0.003 },  // enterprise stable
  { skill: "Git",                 base: 0.10, trend: +0.002 },  // standard tool
  { skill: "Excel",               base: 0.12, trend: -0.002 },  // slowly replaced by BI
  { skill: "Microsoft Office skills", base: 0.10, trend: -0.003 },  // declining relevance
  { skill: "agile methodology",   base: 0.08, trend: +0.001 },  // standard
  { skill: "JIRA",                base: 0.07, trend: +0.001 },  // PM standard
  { skill: "project management",  base: 0.08, trend: +0.000 },  // stable
  { skill: "Linux",               base: 0.07, trend: +0.002 },  // infra stable
  { skill: "C#",                  base: 0.05, trend: +0.000 },  // .NET ecosystem
  { skill: ".NET",                base: 0.04, trend: +0.000 },  // stable
  { skill: "C++",                 base: 0.04, trend: +0.000 },  // systems stable
  { skill: "MongoDB",             base: 0.05, trend: +0.002 },  // NoSQL
  { skill: "Redis",               base: 0.04, trend: +0.003 },  // caching growth
  { skill: "Tableau",             base: 0.05, trend: +0.001 },  // BI stable
  { skill: "Scrum",               base: 0.05, trend: -0.001 },  // agile standard
  { skill: "REST API",            base: 0.07, trend: +0.000 },  // standard
  { skill: "Spring Boot",         base: 0.04, trend: +0.001 },  // Java ecosystem

  // ── DECLINING (negative trend — stronger for visibility) ──
  { skill: "PHP",                 base: 0.08, trend: -0.013 },  // legacy
  { skill: "jQuery",              base: 0.06, trend: -0.015 },  // replaced by React/Vue
  { skill: "Angular",             base: 0.08, trend: -0.011 },  // losing to React
  { skill: "Ruby on Rails",       base: 0.04, trend: -0.008 },  // declining
  { skill: "Hadoop",              base: 0.05, trend: -0.010 },  // replaced by Spark/Databricks
  { skill: "Jenkins",             base: 0.07, trend: -0.013 },  // replaced by GitHub Actions
  { skill: "Puppet",              base: 0.03, trend: -0.007 },  // replaced by Terraform
  { skill: "Chef",                base: 0.03, trend: -0.007 },  // replaced by Terraform
  { skill: "Selenium",            base: 0.06, trend: -0.011 },  // replaced by Playwright/Cypress
  { skill: "manual testing",      base: 0.06, trend: -0.010 },  // automation shift
  { skill: "Webpack",             base: 0.05, trend: -0.011 },  // replaced by Vite
  { skill: "Apache",              base: 0.04, trend: -0.008 },  // replaced by Nginx/cloud
  { skill: "VMware",              base: 0.04, trend: -0.008 },  // cloud migration
  { skill: "TensorFlow",          base: 0.05, trend: -0.008 },  // losing to PyTorch
  { skill: "Sketch",              base: 0.04, trend: -0.009 },  // replaced by Figma
  { skill: "Adobe XD",            base: 0.03, trend: -0.007 },  // replaced by Figma
  { skill: "cold calling",        base: 0.04, trend: -0.008 },  // outbound declining
  { skill: "Laravel",             base: 0.04, trend: -0.008 },  // PHP decline
  { skill: "Nagios",              base: 0.03, trend: -0.007 },  // replaced by Prometheus
];

// ── Generate ~4200 realistic job postings (2022-01 to 2026-03) ──
// Distribution mirrors real-world labor market events + seasonal hiring patterns:
//   • Jan–Mar: peak hiring (new fiscal year, budget releases)
//   • Apr–May: moderate
//   • Jun–Aug: summer dip (vacations, slower hiring)
//   • Sep–Oct: second peak (post-summer push)
//   • Nov–Dec: decline (holidays, year-end freeze)
//
// Year-over-year context:
//   • 2022 (~1000): post-COVID tech boom, strong hiring
//   • 2023 (~1000): Q1–Q2 tech layoffs (Meta/Google/Amazon), H2 recovery
//   • 2024 (~1000): AI hiring wave begins (post-ChatGPT productization)
//   • 2025 (~1000): sustained AI-led growth
//   • 2026 Q1 (200): continued momentum, partial year
function generateDummyData() {
  const rand = seededRandom(42);
  const pick = (arr, n) => {
    const copied = [...arr];
    const result = [];
    for (let i = 0; i < Math.min(n, copied.length); i++) {
      const idx = Math.floor(rand() * copied.length);
      result.push(copied.splice(idx, 1)[0]);
    }
    return result;
  };

  // Build weighted domain selection
  const weightedDomains = [];
  for (const [domain, weight] of Object.entries(DOMAIN_WEIGHTS)) {
    for (let i = 0; i < weight; i++) weightedDomains.push(domain);
  }

  // Monthly distribution with seasonal variation
  // Real pattern: hiring peaks in Jan–Mar and Sep–Oct, dips in Jun–Aug and Dec.
  // Each yearly total ≈ 1000 (2026 partial); shape reflects per-year economic events.
  // Total: ~4200 jobs across 2022-01 to 2026-03 (51 months)
  const monthlyJobs = [
    // 2022 (1000 total) — post-COVID tech boom, peak hiring
    { year: 2022, month: 1, count: 85 }, { year: 2022, month: 2, count: 90 },
    { year: 2022, month: 3, count: 100 }, { year: 2022, month: 4, count: 85 },
    { year: 2022, month: 5, count: 80 }, { year: 2022, month: 6, count: 70 },
    { year: 2022, month: 7, count: 65 }, { year: 2022, month: 8, count: 75 },
    { year: 2022, month: 9, count: 95 }, { year: 2022, month: 10, count: 100 },
    { year: 2022, month: 11, count: 85 }, { year: 2022, month: 12, count: 70 },
    // 2023 (1000 total) — Q1–Q2 tech layoffs, H2 recovery (note Q1 dip vs 2022)
    { year: 2023, month: 1, count: 70 }, { year: 2023, month: 2, count: 65 },
    { year: 2023, month: 3, count: 75 }, { year: 2023, month: 4, count: 80 },
    { year: 2023, month: 5, count: 85 }, { year: 2023, month: 6, count: 80 },
    { year: 2023, month: 7, count: 80 }, { year: 2023, month: 8, count: 85 },
    { year: 2023, month: 9, count: 100 }, { year: 2023, month: 10, count: 105 },
    { year: 2023, month: 11, count: 95 }, { year: 2023, month: 12, count: 80 },
    // 2024 (1000 total) — AI hiring wave, strong Oct peak
    { year: 2024, month: 1, count: 80 }, { year: 2024, month: 2, count: 85 },
    { year: 2024, month: 3, count: 95 }, { year: 2024, month: 4, count: 85 },
    { year: 2024, month: 5, count: 80 }, { year: 2024, month: 6, count: 75 },
    { year: 2024, month: 7, count: 70 }, { year: 2024, month: 8, count: 80 },
    { year: 2024, month: 9, count: 95 }, { year: 2024, month: 10, count: 105 },
    { year: 2024, month: 11, count: 90 }, { year: 2024, month: 12, count: 60 },
    // 2025 (1000 total) — sustained AI-led growth, sharp Dec freeze
    { year: 2025, month: 1, count: 85 }, { year: 2025, month: 2, count: 90 },
    { year: 2025, month: 3, count: 100 }, { year: 2025, month: 4, count: 90 },
    { year: 2025, month: 5, count: 80 }, { year: 2025, month: 6, count: 75 },
    { year: 2025, month: 7, count: 70 }, { year: 2025, month: 8, count: 80 },
    { year: 2025, month: 9, count: 95 }, { year: 2025, month: 10, count: 100 },
    { year: 2025, month: 11, count: 85 }, { year: 2025, month: 12, count: 50 },
    // 2026 Q1 (200 total) — strong year-start, partial year
    { year: 2026, month: 1, count: 60 }, { year: 2026, month: 2, count: 70 },
    { year: 2026, month: 3, count: 70 },
  ];

  // Build flat list of date slots
  const jobDates = [];
  for (const slot of monthlyJobs) {
    const s = new Date(slot.year, slot.month - 1, 1).getTime();
    const e = new Date(slot.year, slot.month, 0).getTime(); // last day of month
    for (let i = 0; i < slot.count; i++) {
      jobDates.push({ s, e, year: slot.year, month: slot.month });
    }
  }

  const data = [];

  for (let i = 0; i < jobDates.length; i++) {
    const { s, e, year: jobYear, month: jobMonth } = jobDates[i];
    const ts = s + Math.floor(rand() * (e - s));
    const posted_date = new Date(ts).toISOString();

    // Year offset for trend calculation (supports fractional for monthly precision)
    const yearOffset = (jobYear - 2022) + (jobMonth - 1) / 12;

    // Pick domain — shift domain weights over time (more data/AI jobs in later years)
    let domain;
    if (yearOffset > 2 && rand() < 0.08) {
      // Post-2024: extra 8% chance of data/AI jobs
      domain = rand() < 0.6 ? "data" : "security";
    } else {
      domain = weightedDomains[Math.floor(rand() * weightedDomains.length)];
    }

    const roles = ROLES[domain];
    const baseRole = roles[Math.floor(rand() * roles.length)];

    // Title with level — allow duplicates (real world: many "Software Engineer" posts)
    const level = LEVELS[Math.floor(rand() * LEVELS.length)];
    const title = `${level}${baseRole}`.trim();

    // Pick domain-specific skills (4–9, slightly more in recent years)
    const domainSkillPool = DOMAIN_SKILLS[domain] || DOMAIN_SKILLS.engineering;
    const allDomainSkills = Object.values(domainSkillPool).flat();
    const extraSkills = yearOffset > 2 ? 1 : 0; // jobs require more skills over time
    const primaryCount = 4 + extraSkills + Math.floor(rand() * 5);
    const skillSet = new Set(pick(allDomainSkills, primaryCount));

    // Soft skills (0–3)
    const softCount = rand() < 0.25 ? 0 : 1 + Math.floor(rand() * 3);
    for (const s of pick(UNIVERSAL_SOFT, softCount)) skillSet.add(s);

    // Language (35% chance, higher for non-tech domains)
    const langChance = ["hr", "sales", "support", "marketing"].includes(domain) ? 0.5 : 0.35;
    if (rand() < langChance) {
      for (const s of pick(LANGUAGES, 1)) skillSet.add(s);
    }

    // ── Apply time-dependent skill trends ──
    // Uses fractional yearOffset for monthly granularity.
    // AI boom skills (LLM, RAG, etc.) don't appear until Nov 2022 (ChatGPT launch).
    for (const { skill, base, trend } of SKILL_TRENDS) {
      if (skillSet.has(skill)) continue;
      const isAiBoom = AI_BOOM_SKILLS.has(skill);
      const effectiveOffset = isAiBoom
        ? Math.max(0, yearOffset - AI_BOOM_START)
        : yearOffset;
      const prob = Math.max(0, Math.min(0.5, base + trend * effectiveOffset));
      if (rand() < prob) {
        skillSet.add(skill);
      }
    }

    data.push({ job_title: title, skills: [...skillSet], posted_date });
  }

  // Sort by date
  data.sort((a, b) => new Date(a.posted_date) - new Date(b.posted_date));

  return data;
}

export const dummy_data = generateDummyData();
