import random
import re
from collections import defaultdict, Counter

# ----- DATE VALIDATION FUNCTION -----
def is_valid_date(date_str):
    """
    Validates a date string in DD/MM/YYYY format.
    Returns True if valid, False otherwise.
    """
    # Check format (DD/MM/YYYY)
    if not re.match(r"^\d{2}/\d{2}/\d{4}$", date_str):
        return False
    
    day_str, month_str, year_str = date_str.split("/")
    try:
        day = int(day_str)
        month = int(month_str)
        year = int(year_str)
    except ValueError:
        return False  # Non-integer values
    
    # Validate year range
    if year < 0 or year > 9999:
        return False
    
    # Validate month
    if month < 1 or month > 12:
        return False
    
    # Validate day
    if day < 1:
        return False
    
    # Days per month logic
    if month in [4, 6, 9, 11] and day > 30:
        return False  # 30-day months
    elif month == 2:
        # Leap year check
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        max_day = 29 if is_leap else 28
        if day > max_day:
            return False
    elif day > 31:
        return False  # 31-day months
    
    return True

# ----- GENETIC ALGORITHM COMPONENTS -----

def categorize_date(date_tuple):
    """
    Categorizes a date tuple (day, month, year) into a specific category.
    Returns a tuple of (validity, category, subcategory).
    """
    day, month, year = date_tuple
    date_str = f"{day:02d}/{month:02d}/{year:04d}" # Convert to string for validation
    
    # Check validity first
    valid = is_valid_date(date_str)
    
    # Categorize based on validity
    if valid:
        # Determine specific valid category
        if month == 2:
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) # Leap year check
            if is_leap:
                return (True, "Valid", "Leap Year February")
            else:
                return (True, "Valid", "Non-Leap February")
        elif month in [4, 6, 9, 11]:  # 30-day months
            return (True, "Valid", "30-Day Month")
        else:
            return (True, "Valid", "31-Day Month")
    else:
        # Determine specific invalid category
        if month < 1 or month > 12:   # Invalid month
            return (False, "Invalid", "Invalid Month")
        elif day < 1:
            return (False, "Invalid", "Day < 1")
        elif month in [4, 6, 9, 11] and day > 30:    # 30-day months
            return (False, "Invalid", f"Day > 30 in {month}-month")
        elif month == 2:
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)  # Leap year check
            max_day = 29 if is_leap else 28
            if day > max_day:
                if is_leap:
                    return (False, "Invalid", "Day > 29 in February (Leap)")
                else:
                    return (False, "Invalid", "Day > 28 in February (Non-Leap)")
        elif day > 31:
            return (False, "Invalid", "Day > 31")
    
    # This should never happen, but just in case
    return (False, "Invalid", "Unknown")

def is_boundary_case(date_tuple):
    """
    Determines if a date tuple is a boundary case.
    Returns (is_boundary, boundary_type) tuple.
    """
    day, month, year = date_tuple
    
    # Check for year boundaries
    if year == 0 and month == 1 and day == 1:   # Min date
        return (True, "Min Date")
    elif year == 9999 and month == 12 and day == 31:   # Max date
        return (True, "Max Date")
    
    # Check for leap year Feb 29
    if month == 2 and day == 29:   # Leap year Feb 29
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)   # Leap year check
        if is_leap:
            return (True, "Leap Year Boundary")
    
    # Check for specific month-end boundaries
    if (month in [1, 3, 5, 7, 8, 10, 12] and day == 31) or (month in [4, 6, 9, 11] and day == 30):    # End of month
        return (True, f"End of Month ({month})")
    
    # Not a boundary case
    return (False, "")

def initialize_population(size=100):
    """
    Initialize a population of random date tuples.
    Each date is a tuple of (day, month, year).
    """
    population = []
    
    # Make sure we include some boundary values
    boundary_cases = [
        (1, 1, 0),      # Min date
        (31, 12, 9999), # Max date
        (29, 2, 2020),  # Leap year
        (29, 2, 2021),  # Invalid leap year
        (31, 4, 2023),  # Invalid day for April
    ]
    
    # Add boundary cases to the population
    population.extend(boundary_cases)
    
    # Generate the rest randomly
    for _ in range(size - len(boundary_cases)):
        day = random.randint(1, 31)   # Random day
        month = random.randint(1, 12)    # Random month
        year = random.randint(0, 9999)    # Random year
        population.append((day, month, year))
    
    return population

def calculate_fitness(chromosome, population, covered_categories):
    """
    Calculate fitness score for a chromosome based on category coverage.
    - Higher score for covering new categories
    - Penalty for redundant test cases
    """
    # Get category for this test case
    date_tuple = chromosome
    category_data = categorize_date(date_tuple)
    boundary_data = is_boundary_case(date_tuple)
    
    # Calculate base fitness score
    fitness = 1.0
    
    # Bonus for covering new categories
    full_category = (category_data, boundary_data[0])
    if full_category not in covered_categories:
        fitness += 10.0  # Big bonus for new categories
    
    # Bonus for boundary cases
    if boundary_data[0]:
        fitness += 5.0
    
    # Penalty for redundancy (how many similar test cases exist)
    redundant_count = 0
    for other in population:
        if other != chromosome:
            other_category = categorize_date(other)
            other_boundary = is_boundary_case(other)
            if other_category == category_data and other_boundary[0] == boundary_data[0]:
                redundant_count += 1
    
    # Apply redundancy penalty
    if redundant_count > 0:
        fitness /= (1 + redundant_count * 0.5)  # Diminishing returns for redundant cases
    
    return fitness

def selection(population, fitness_scores):
    """
    Rank-based selection that prioritizes high-fitness chromosomes.
    Returns a selected parent.
    """
    # Rank-based selection
    population_with_fitness = list(zip(population, 
                                       fitness_scores))  # Combine population with fitness
    population_with_fitness.sort(key=lambda x: x[1], 
                                 reverse=True)   # Sort by fitness (higher is better)
    
    # Use weighted random selection based on rank
    ranks = list(range(1, len(population_with_fitness) + 1))
    weights = [1/r for r in ranks]  # Higher ranks (lower values) get higher weights
    total_weight = sum(weights)  # Normalize weights
    normalized_weights = [w/total_weight for w in weights]   # Normalize weights
    
    # Select based on normalized weights
    selected_idx = random.choices(range(len(population_with_fitness)), 
                                  weights=normalized_weights, k=1)[0]   # Select one index
    return population_with_fitness[selected_idx][0]

def crossover(parent1, parent2):
    """
    Perform crossover between two parent chromosomes.
    Each child inherits different components from different parents.
    """
    # Extract components
    day1, month1, year1 = parent1
    day2, month2, year2 = parent2
    
    # Randomly determine crossover type
    crossover_type = random.randint(1, 3)  # 1, 2, or 3
    
    if crossover_type == 1:
        # Child inherits day from parent1, month/year from parent2
        child = (day1, month2, year2)
    elif crossover_type == 2:
        # Child inherits month from parent1, day/year from parent2
        child = (day2, month1, year2)
    else:
        # Child inherits year from parent1, day/month from parent2
        child = (day2, month2, year1)
    
    return child

def mutation(chromosome, mutation_rate=0.15):
    """
    Mutate a chromosome with a given probability.
    Can change day, month, or year by a small random amount.
    """
    day, month, year = chromosome
    
    # Decide if we're mutating this chromosome
    if random.random() <= mutation_rate:
        # Decide which component to mutate
        component = random.choice(['day', 'month', 'year'])
        
        if component == 'day':
            # Mutate day with small adjustments
            day_change = random.choice([-3, -2, -1, 1, 2, 3])  # Small random change
            day = max(1, min(31, day + day_change))  # Keep within reasonable bounds
        elif component == 'month':
            # Mutate month with small adjustments
            month_change = random.choice([-1, 1])
            month = max(1, min(12, month + month_change))  # Keep within reasonable bounds
        else:
            # Mutate year with bigger adjustments
            if random.random() < 0.5:  # Small year change
                # Small year change
                year_change = random.choice([-10, -5, -1, 1, 5, 10])   # Small random change
                year = max(0, min(9999, year + year_change))   # Keep within reasonable bounds
            else:
                # Large year change for more diversity
                year_change = random.choice([-1000, -500, -100, 100, 500, 1000])   # Large random change
                year = max(0, min(9999, year + year_change))   # Keep within reasonable bounds
    
    return (day, month, year)

def get_coverage_percentage(test_cases):
    """
    Calculate the coverage percentage of the test cases.
    """
    # All possible valid categories
    valid_categories = [
        ((True, "Valid", "Leap Year February"), False),
        ((True, "Valid", "Non-Leap February"), False),
        ((True, "Valid", "30-Day Month"), False),
        ((True, "Valid", "31-Day Month"), False),
        ((True, "Valid", "Leap Year February"), True),   # Boundary leap year
        ((True, "Valid", "31-Day Month"), True),         # Boundary end of month
        ((True, "Valid", "30-Day Month"), True),         # Boundary end of month
    ]
    
    # All possible invalid categories
    invalid_categories = [
        ((False, "Invalid", "Invalid Month"), False),
        ((False, "Invalid", "Day < 1"), False),
        ((False, "Invalid", "Day > 30 in 4-month"), False),
        ((False, "Invalid", "Day > 30 in 6-month"), False),
        ((False, "Invalid", "Day > 30 in 9-month"), False),
        ((False, "Invalid", "Day > 30 in 11-month"), False),
        ((False, "Invalid", "Day > 28 in February (Non-Leap)"), False),
        ((False, "Invalid", "Day > 29 in February (Leap)"), False),
        ((False, "Invalid", "Day > 31"), False),
    ]
    
    # Boundary specific categories
    boundary_categories = [
        ((True, "Valid", "31-Day Month"), True, "Min Date"),
        ((True, "Valid", "31-Day Month"), True, "Max Date"),
    ]
    
    # Count how many categories we've covered
    covered_valid = 0
    covered_invalid = 0
    covered_boundary = 0
    
    # Track covered category types
    covered_types = set()
    
    # Check coverage
    for date_tuple in test_cases:
        category = categorize_date(date_tuple)
        boundary, boundary_type = is_boundary_case(date_tuple)
        
        # Check if this covers a valid category
        for valid_cat in valid_categories:
            if (category, boundary) == valid_cat and valid_cat not in covered_types:
                covered_valid += 1
                covered_types.add(valid_cat)
        
        # Check if this covers an invalid category
        for invalid_cat in invalid_categories:
            if (category, boundary) == invalid_cat and invalid_cat not in covered_types:
                covered_invalid += 1
                covered_types.add(invalid_cat)
        
        # Check boundary specific conditions
        for bound_cat in boundary_categories:
            cat, is_bound, bound_type = bound_cat
            if (category, boundary) == (cat, is_bound) and boundary_type == bound_type:
                covered_boundary += 1
                covered_types.add(bound_cat)
    
    # Calculate coverage percentage
    total_categories = len(valid_categories) + len(invalid_categories) + len(boundary_categories)
    covered = covered_valid + covered_invalid + covered_boundary
    coverage_percentage = (covered / total_categories) * 100
    
    return coverage_percentage

# ----- LOCAL SEARCH ALGORITHM -----
def get_all_possible_categories():
    """
    Get all possible test case categories for coverage analysis.
    Returns a list of categories that should be covered.
    """
    # All possible valid categories
    valid_categories = [
        ((True, "Valid", "Leap Year February"), False),
        ((True, "Valid", "Non-Leap February"), False),
        ((True, "Valid", "30-Day Month"), False),
        ((True, "Valid", "31-Day Month"), False),
        ((True, "Valid", "Leap Year February"), True),   # Boundary leap year
        ((True, "Valid", "31-Day Month"), True),         # Boundary end of month
        ((True, "Valid", "30-Day Month"), True),         # Boundary end of month
    ]
    
    # All possible invalid categories
    invalid_categories = [
        ((False, "Invalid", "Invalid Month"), False),
        ((False, "Invalid", "Day < 1"), False),
        ((False, "Invalid", "Day > 30 in 4-month"), False),
        ((False, "Invalid", "Day > 30 in 6-month"), False),
        ((False, "Invalid", "Day > 30 in 9-month"), False),
        ((False, "Invalid", "Day > 30 in 11-month"), False),
        ((False, "Invalid", "Day > 28 in February (Non-Leap)"), False),
        ((False, "Invalid", "Day > 29 in February (Leap)"), False),
        ((False, "Invalid", "Day > 31"), False),
    ]
    
    # Boundary specific categories
    boundary_categories = [
        ((True, "Valid", "31-Day Month"), True, "Min Date"),
        ((True, "Valid", "31-Day Month"), True, "Max Date"),
    ]
    
    # Combine all categories
    all_categories = []
    all_categories.extend([(cat, "Valid") for cat in valid_categories])
    all_categories.extend([(cat, "Invalid") for cat in invalid_categories])
    all_categories.extend([(cat, "Boundary") for cat in boundary_categories])
    
    return all_categories

def get_missing_categories(population):
    """
    Identify categories not covered by the current population.
    Returns a list of missing categories.
    """
    all_categories = get_all_possible_categories()
    covered_categories = set()
    
    # Find categories covered by the population
    for date_tuple in population:
        category = categorize_date(date_tuple)
        boundary, boundary_type = is_boundary_case(date_tuple)
        
        # Handle regular categories
        full_category = (category, boundary)
        if full_category in [(cat[0], "Valid") for cat in all_categories] or \
           full_category in [(cat[0], "Invalid") for cat in all_categories]:
            covered_categories.add(full_category)
        
        # Handle boundary-specific categories
        for cat, cat_type in all_categories:
            if cat_type == "Boundary" and len(cat) == 3:
                cat_data, is_bound, bound_type = cat
                if (category, boundary) == (cat_data, is_bound) and boundary_type == bound_type:
                    covered_categories.add((cat, cat_type))
    
    # Find missing categories
    missing_categories = [cat for cat in all_categories if cat not in covered_categories]
    return missing_categories

def generate_test_for_category(category_info):
    """
    Generate a specific test case for a given category.
    Returns a date tuple (day, month, year) targeting the category.
    """
    category, category_type = category_info
    
    if category_type == "Valid":
        validity, _, subcat = category[0]
        is_boundary = category[1]
        
        if subcat == "Leap Year February":
            # Generate a valid leap year February date
            year = 2020  # Known leap year
            if is_boundary:
                return (29, 2, year)  # February 29 (boundary)
            else:
                return (random.randint(1, 28), 2, year)  # Random February day
        
        elif subcat == "Non-Leap February":
            # Generate a valid non-leap year February date
            year = 2021  # Known non-leap year
            return (random.randint(1, 28), 2, year)  # Random February day
        
        elif subcat == "30-Day Month":
            # Generate a valid 30-day month date
            month = random.choice([4, 6, 9, 11])
            if is_boundary:
                return (30, month, random.randint(1, 9999))  # End of 30-day month
            else:
                return (random.randint(1, 29), month, random.randint(1, 9999))
        
        elif subcat == "31-Day Month":
            # Generate a valid 31-day month date
            month = random.choice([1, 3, 5, 7, 8, 10, 12])
            if is_boundary:
                return (31, month, random.randint(1, 9999))  # End of 31-day month
            else:
                return (random.randint(1, 30), month, random.randint(1, 9999))
    
    elif category_type == "Invalid":
        _, _, subcat = category[0]
        
        if subcat == "Invalid Month":
            # Generate a date with invalid month
            return (random.randint(1, 28), random.choice([0, 13]), random.randint(1, 9999))
        
        elif subcat == "Day < 1":
            # Generate a date with day < 1
            return (0, random.randint(1, 12), random.randint(1, 9999))
        
        elif subcat.startswith("Day > 30 in"):
            # Generate a date with day > 30 in a 30-day month
            month_num = int(subcat.split("-")[0].split(" ")[-1])
            return (31, month_num, random.randint(1, 9999))
        
        elif subcat == "Day > 28 in February (Non-Leap)":
            # Generate a date with day > 28 in February of non-leap year
            year = 2021  # Known non-leap year
            return (29, 2, year)
        
        elif subcat == "Day > 29 in February (Leap)":
            # Generate a date with day > 29 in February of leap year
            year = 2020  # Known leap year
            return (30, 2, year)
        
        elif subcat == "Day > 31":
            # Generate a date with day > 31
            return (32, random.randint(1, 12), random.randint(1, 9999))
    
    elif category_type == "Boundary":
        cat_data, is_bound, bound_type = category
        
        if bound_type == "Min Date":
            return (1, 1, 0)  # Minimum possible date
        
        elif bound_type == "Max Date":
            return (31, 12, 9999)  # Maximum possible date
    
    # Default fallback - return a random date
    return (random.randint(1, 28), random.randint(1, 12), random.randint(1, 9999))

def apply_local_search(population, iterations=50):
    """
    Apply local search to improve test coverage.
    Returns an improved population with better coverage.
    """
    # Get initial coverage
    initial_coverage = get_coverage_percentage(population)
    print(f"Initial coverage before local search: {initial_coverage:.2f}%")
    
    # Make a copy of the population to modify
    improved_population = population.copy()
    
    # Track seen categories
    covered_categories = set()
    for date_tuple in improved_population:
        category = categorize_date(date_tuple)
        boundary, boundary_type = is_boundary_case(date_tuple)
        covered_categories.add((category, boundary))
    
    # Iterate to improve population
    for iteration in range(iterations):
        # Find missing categories
        missing_categories = get_missing_categories(improved_population)
        
        if not missing_categories:
            print(f"All categories covered after {iteration} iterations!")
            break
        
        # Try to generate test cases for missing categories
        for category_info in missing_categories[:5]:  # Process up to 5 missing categories per iteration
            new_test_case = generate_test_for_category(category_info)
            
            # Add to population, potentially replacing a redundant test case
            if len(improved_population) >= 100:
                # Find redundant test cases
                category_counts = defaultdict(int)
                for date_tuple in improved_population:
                    cat = categorize_date(date_tuple)
                    bound = is_boundary_case(date_tuple)
                    category_counts[(cat, bound[0])] += 1
                
                # Find most redundant category
                most_redundant = max(category_counts.items(), key=lambda x: x[1])
                
                if most_redundant[1] > 1:
                    # Replace one of the redundant test cases
                    for i, date_tuple in enumerate(improved_population):
                        cat = categorize_date(date_tuple)
                        bound = is_boundary_case(date_tuple)
                        if (cat, bound[0]) == most_redundant[0]:
                            improved_population[i] = new_test_case
                            break
                else:
                    # If no redundancies, replace a random test case
                    random_idx = random.randint(0, len(improved_population) - 1)
                    improved_population[random_idx] = new_test_case
            else:
                # If population is small, just add the new test case
                improved_population.append(new_test_case)
        
        # Every 10 iterations, check progress
        if iteration % 10 == 0 and iteration > 0:
            current_coverage = get_coverage_percentage(improved_population)
            print(f"Iteration {iteration}: Coverage = {current_coverage:.2f}%")
    
    # Calculate final coverage
    final_coverage = get_coverage_percentage(improved_population)
    print(f"Final coverage after local search: {final_coverage:.2f}%")
    print(f"Coverage improvement: {final_coverage - initial_coverage:.2f}%")
    
    return improved_population

def run_genetic_algorithm(population_size=50, generations=100, mutation_rate=0.15, target_coverage=95, apply_local=True):
    """
    Run the genetic algorithm to evolve test cases.
    """
    # Initialize population
    population = initialize_population(population_size)
    
    # Track coverage categories
    covered_categories = set()
    
    # Track best solution
    best_solution = []
    best_coverage = 0
    
    # Generation loop
    for generation in range(generations):
        # Calculate fitness for each chromosome
        fitness_scores = [calculate_fitness(chrom, population, covered_categories) for chrom in population]
        
        # Update covered categories
        for chrom in population:
            category_data = categorize_date(chrom)
            boundary_data = is_boundary_case(chrom)
            covered_categories.add((category_data, boundary_data[0]))
        
        # Create a new population
        new_population = []
        
        # Elitism: keep the best solutions
        elite_count = max(1, int(population_size * 0.1))
        population_with_fitness = list(zip(population, 
                                           fitness_scores))  # Combine population with fitness
        population_with_fitness.sort(key=lambda x: x[1], 
                                     reverse=True)  # Sort by fitness (higher is better)
        elites = [p[0] for p in population_with_fitness[:elite_count]]
        new_population.extend(elites)
        
        # Fill the rest with offspring
        while len(new_population) < population_size:
            # Selection
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutation(child, mutation_rate)
            
            # Add to new population
            new_population.append(child)
        
        # Update population
        population = new_population
        
        # Check current test case coverage
        current_coverage = get_coverage_percentage(population)
        
        # Update best solution if better
        if current_coverage > best_coverage:
            best_coverage = current_coverage
            
            # Convert to nicely formatted test cases
            valid_cases = []
            invalid_cases = []
            boundary_cases = []
            
            for date_tuple in population:
                day, month, year = date_tuple
                date_str = f"{day:02d}/{month:02d}/{year:04d}"  # Convert to string
                category = categorize_date(date_tuple)
                boundary, boundary_type = is_boundary_case(date_tuple)
                
                if boundary:
                    boundary_cases.append((date_str, boundary_type))
                elif category[0]:  # Valid
                    valid_cases.append((date_str, category[2]))
                else:  # Invalid
                    invalid_cases.append((date_str, category[2]))
            
            # Create the best solution with the most diverse test cases
            best_solution = {
                'Valid': valid_cases[:10],  # Limit to 10 valid
                'Invalid': invalid_cases[:10],  # Limit to 10 invalid
                'Boundary': boundary_cases[:5]  # Limit to 5 boundary
            }
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Coverage = {current_coverage:.2f}%")
        
        # Check if we've reached target coverage
        if current_coverage >= target_coverage:
            print(f"Target coverage achieved in generation {generation}")
            break
    
    # Apply local search if enabled
    if apply_local:
        print("\nApplying local search to improve coverage...")
        improved_population = apply_local_search(population)
        
        # Calculate coverage after local search
        local_coverage = get_coverage_percentage(improved_population)
        
        # If local search improved coverage, update the best solution
        if local_coverage > best_coverage:
            best_coverage = local_coverage
            
            # Convert to nicely formatted test cases
            valid_cases = []
            invalid_cases = []
            boundary_cases = []
            
            for date_tuple in improved_population:
                day, month, year = date_tuple
                date_str = f"{day:02d}/{month:02d}/{year:04d}"  # Convert to string
                category = categorize_date(date_tuple)
                boundary, boundary_type = is_boundary_case(date_tuple)
                
                if boundary:
                    boundary_cases.append((date_str, boundary_type))
                elif category[0]:  # Valid
                    valid_cases.append((date_str, category[2]))
                else:  # Invalid
                    invalid_cases.append((date_str, category[2]))
            
            # Create the best solution with the most diverse test cases
            best_solution = {
                'Valid': valid_cases[:10],  # Limit to 10 valid
                'Invalid': invalid_cases[:10],  # Limit to 10 invalid
                'Boundary': boundary_cases[:5]  # Limit to 5 boundary
            }
    
    # Return the best solution and the final coverage
    return best_solution, best_coverage, generation + 1

# ----- MAIN EXECUTION -----
def main():
    print("Starting Genetic Algorithm for Date Test Case Generation")
    
    # Run the standard GA for comparison
    print("\n=== STANDARD GENETIC ALGORITHM ===")
    standard_solution, standard_coverage, standard_generations = run_genetic_algorithm(
        population_size=100,
        generations=150,
        mutation_rate=0.15,
        target_coverage=95,
        apply_local=False
    )
    
    print(f"\nStandard GA Results:")
    print(f"Coverage: {standard_coverage:.2f}%")
    print(f"Generations needed: {standard_generations}")
    
    # Run GA with local search
    print("\n=== GENETIC ALGORITHM WITH LOCAL SEARCH ===")
    hybrid_solution, hybrid_coverage, hybrid_generations = run_genetic_algorithm(
        population_size=100,
        generations=150,
        mutation_rate=0.15,
        target_coverage=95,
        apply_local=True
    )
    
    print(f"\nHybrid GA Results:")
    print(f"Coverage: {hybrid_coverage:.2f}%")
    print(f"Generations needed: {hybrid_generations}")
    
    # Print the best test cases found
    print("\n=== BEST TEST CASES ===")
    print("Valid Test Cases:")
    for date_str, category in hybrid_solution['Valid']:
        print(f"  {date_str} - {category}")
    
    print("\nInvalid Test Cases:")
    for date_str, category in hybrid_solution['Invalid']:
        print(f"  {date_str} - {category}")
    
    print("\nBoundary Test Cases:")
    for date_str, category in hybrid_solution['Boundary']:
        print(f"  {date_str} - {category}")
    
    # Compare results
    print("\n=== COMPARISON ===")
    print(f"Standard GA: {standard_coverage:.2f}% coverage in {standard_generations} generations")
    print(f"Hybrid GA: {hybrid_coverage:.2f}% coverage in {hybrid_generations} generations")
    
if __name__ == "__main__":
    main()