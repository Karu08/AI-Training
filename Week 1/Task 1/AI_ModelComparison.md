AI Model Comparison Sheet

&nbsp;

**Department Use Cases**

\- AppDev (Code Generation, Quality)  

\- Data Analysis & SQL (Data)  

\- Infra Automation (DevOps)  

\- Ease of Use, and Speed/Latency

&nbsp;

**Models Evaluated**

\- GPT-5  

\- Claude 4.5

\- Gemini Flash  

\- DeepSeek-R1:7B (Ollama)  

&nbsp;

**Rating Legend**

\- excellent

\- good

\- basic / limited support

\- not supported

&nbsp;

**Comparison Table**

| **Criteria** | **Example Prompt (Retail Banking)** | **GPT-4o** | **Claude Sonnet** | **Gemini Flash** | **DeepSeek-R1:7B (Ollama)** | **Comments** |
| --- | --- | --- | --- | --- | --- | --- |
| Code Quality (AppDev) | Design REST APIs for retail banking accounts with deposit and withdrawal logic, including validations. | Excellent | Excellent | Good | Basic / Limited support | GPT-4o and Claude generate clean, structured APIs. Claude excels in readable business logic. Gemini is usable but sometimes inconsistent. DeepSeek handles only simple CRUD. |
| Business Logic Reasoning | Implement withdrawal logic ensuring balance checks, overdraft limits, and error handling. | Excellent | Excellent | Good | Basic / Limited support | GPT-4o and Claude correctly handle edge cases like insufficient balance and limits. DeepSeek often misses edge conditions. |
| SQL Generation (Data) | Write an SQL query to generate monthly statements showing total deposits and withdrawals per customer. | Excellent | Excellent | Good | Basic / Limited support | GPT-4o and Claude generate accurate joins and aggregations. Gemini may produce inefficient SQL. DeepSeek struggles with complex joins. |
| Infrastructure Automation (Scripts) | Write Terraform code to deploy a retail banking backend application and a database. | Excellent | Good | Good | Basic / Limited support | GPT-4o produces production-ready Terraform and scripts. Claude is conservative. Gemini works well for YAML. DeepSeek requires heavy fixes. |
| Ease of Use | Improve the previous solution by following banking best practices and adding comments. | Excellent | Excellent | Good | Basic / Limited support | GPT-4o and Claude need minimal prompt tuning. Gemini needs clearer instructions. DeepSeek requires step-by-step prompts. |
| Speed / Latency | Quickly generate a CI/CD pipeline for a retail banking application. | Good | Good | Excellent | Excellent | Gemini Flash responds fastest among cloud models. DeepSeek is fast locally but lower quality. |

**Summary**

**Primary Model: GPT-4o →** Best overall choice for retail banking applications across **App Development, Data (SQL), and DevOps/Infrastructure Automation**

**Secondary Model: Claude Sonnet →** Excellent for **clear business logic, readable code, and documentation**

**Fast / Lightweight Option: Gemini Flash →** Suitable for **quick SQL queries, CI/CD drafts, and rapid prototyping**

**Local / On-Prem Option: DeepSeek-R1:7B (Ollama) →** Useful only for **basic, non-critical tasks in offline environments**