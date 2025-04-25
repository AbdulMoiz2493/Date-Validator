# 🧬 Date Validator – Test Case Generator using Genetic Algorithm & Local Search

This project implements an **automated test case generator** for a **Date Validation Function**, using a **Genetic Algorithm (GA)** combined with a **Local Search algorithm** to evolve high-coverage test suites.  
The goal is to validate dates in `DD/MM/YYYY` format by exploring both **valid** and **invalid** edge cases with high precision.

---

## 📌 Objective

Automatically evolve intelligent date strings that:
- ✅ Validate real calendar dates (e.g., 15/05/2023)
- ❌ Capture invalid cases (e.g., 29/02/2021, 32/01/2023)
- 🎯 Include critical boundaries (e.g., 01/01/0000, 31/12/9999)

---

## ⚙️ Genetic Algorithm Design

### 🧬 Chromosome Representation
Each chromosome is a tuple `(day, month, year)` converted into a string:  
📅 `DD/MM/YYYY`

### 🌱 Population Initialization
Random dates generated with:
- Day: `1–31`
- Month: `1–12`
- Year: `0000–9999`

### 🧠 Fitness Function
Each date earns a fitness score based on:
- ✅ Coverage of unique categories (leap years, 30/31-day months, invalid formats, etc.)
- ❌ Penalization of redundant or repeated test cases

### 🎯 Selection
- Rank-based selection strategy

### 🔄 Crossover
- Swap `day`, `month`, and `year` segments between two parents

### 🔀 Mutation
- 15% chance:
  - Day ±3
  - Month ±1
  - Year ±100

### 🛑 Termination
- Max 100 generations or 95% category coverage

---

## 🧭 Local Search Optimization (BONUS ✅)

After GA evolution:
- A **hill-climbing local search** refines the top-performing test cases
- Explores neighbors to further **maximize unique category coverage**
- Results in a stronger and more diverse final test suite

---

## 🧪 Target Test Categories

### ✅ Valid
- Leap year dates: `29/02/2020`
- 30-day months: `30/06/2023`
- 31-day months: `31/01/2023`

### ❌ Invalid
- Day > 31: `32/01/2023`
- Month > 12: `15/13/2023`
- Invalid leap years: `29/02/2021`, `29/02/1900`

### 🧱 Boundary
- Minimum Year: `01/01/0000`
- Maximum Year: `31/12/9999`

---

## 📦 Output

- ✅ Top evolved test cases (CSV & JSON)
- 📊 Graph showing coverage improvement across generations
- 📁 Summary of test case categories and total coverage

---

## 🧪 Example Output

| Test Case    | Category                |
|--------------|--------------------------|
| 29/02/2020   | ✅ Valid - Leap Year      |
| 31/04/2023   | ❌ Invalid - April 31     |
| 32/01/2022   | ❌ Invalid - Day > 31     |
| 15/13/2023   | ❌ Invalid - Month > 12   |
| 01/01/0000   | 🧱 Boundary - Min Year     |
| 31/12/9999   | 🧱 Boundary - Max Year     |

---

## 📊 Report Includes

- 🧠 Design of fitness function and categories
- ⚙️ Mutation/crossover tuning analysis
- 📈 Line graph of generation-wise coverage
- 🆚 GA vs. Random Testing
- ✅ Final test case suite (CSV/JSON)
- 🔍 Impact of local search on refinement

---

## 📁 Project Structure

```
📂 Date-Validator/
├── Date-Validator.pdf           # Report explaining the implementation
├── Date-Validator.py            # main python program
├── Test-Cases.csv               # test cases for date validator


---

## 🚀 How to Run

### Requirements
- Python 3.x
- Libraries: `random`, `numpy`, `csv`, `matplotlib`, `json`

### Execution

```bash
git clone https://github.com/AbdulMoiz2493/Date-Validator.git
cd Date-Validator
python Date-Validator.py
```

Results:
- 🧪 `test_cases.csv` – Evolved test cases with labels
- 📈 `coverage_graph.png` – Performance of the algorithm

---

## 👨‍💻 Author

**Abdul Moiz**  
📧 abdulmoiz8895@gmail.com  
📍 FAST NUCES Islamabad  
🌐 [GitHub: AbdulMoiz2493](https://github.com/AbdulMoiz2493)

---

## 📝 License

MIT License – Free to use, modify, and share.

---

## 🌟 Star the Repo

If you found this project helpful, don’t forget to ⭐ [star the repository](https://github.com/AbdulMoiz2493/Date-Validator) and share it with fellow testers and developers!
```
