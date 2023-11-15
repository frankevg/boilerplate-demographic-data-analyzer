import pandas as pd


def calculate_demographic_data(print_data=True):
  # Load the dataset
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
  column_names = [
      "age", "workclass", "fnlwgt", "education", "education-num",
      "marital-status", "occupation", "relationship", "race", "sex",
      "capital-gain", "capital-loss", "hours-per-week", "native-country",
      "salary"
  ]
  df = pd.read_csv(url,
                   header=None,
                   names=column_names,
                   engine='python',
                   delimiter=',\s',
                   na_values="?")

  # 1. How many people of each race are represented in this dataset?
  race_count = df['race'].value_counts()

  # 2. What is the average age of men?
  average_age_men = df[df['sex'] == 'Male']['age'].mean()

  # 3. What is the percentage of people who have a Bachelor's degree?
  percentage_bachelors = (df['education'] == 'Bachelors').sum() / len(df) * 100

  # 4. What percentage of people with advanced education make more than 50K?
  higher_education = df[df['education'].isin(
      ['Bachelors', 'Masters', 'Doctorate'])]
  higher_education_rich = (higher_education['salary']
                           == '>50K').sum() / len(higher_education) * 100

  # 5. What percentage of people without advanced education make more than 50K?
  lower_education = df[~df['education'].
                       isin(['Bachelors', 'Masters', 'Doctorate'])]
  lower_education_rich = (lower_education['salary']
                          == '>50K').sum() / len(lower_education) * 100

  # 6. What is the minimum number of hours a person works per week?
  min_work_hours = df['hours-per-week'].min()

  # 7. What percentage of the people who work the minimum number of hours per week have a salary of more than 50K?
  num_min_workers = len(df[df['hours-per-week'] == min_work_hours])
  rich_percentage = (df[(df['hours-per-week'] == min_work_hours) &
                        (df['salary'] == '>50K')].shape[0] /
                     num_min_workers) * 100

  # 8. What country has the highest percentage of people that earn >50K?
  highest_earning_country_percentage = df[df['salary'] == '>50K'][
      'native-country'].value_counts() / df['native-country'].value_counts()
  highest_earning_country = highest_earning_country_percentage.idxmax()
  highest_earning_country_percentage = highest_earning_country_percentage.max(
  ) * 100

  # 9. Identify the most popular occupation for those who earn >50K in India.
  top_IN_occupation = df[(df['native-country'] == 'India') & (
      df['salary'] == '>50K')]['occupation'].value_counts().idxmax()

  # Results dictionary with rounded values
  results = {
      'race_count':
      race_count,
      'average_age_men':
      round(average_age_men, 1),
      'percentage_bachelors':
      round(percentage_bachelors, 1),
      'higher_education_rich':
      round(higher_education_rich, 1),
      'lower_education_rich':
      round(lower_education_rich, 1),
      'min_work_hours':
      min_work_hours,
      'rich_percentage':
      round(rich_percentage, 1),
      'highest_earning_country':
      highest_earning_country,
      'highest_earning_country_percentage':
      round(highest_earning_country_percentage, 1),
      'top_IN_occupation':
      top_IN_occupation
  }

  # Display the results
  if print_data:
    print("Number of each race:\n", results['race_count'])
    print("Average age of men:", results['average_age_men'])
    print(
        f"Percentage with Bachelors degrees: {results['percentage_bachelors']:.1f}%"
    )
    print(
        f"Percentage with higher education that earn >50K: {results['higher_education_rich']:.1f}%"
    )
    print(
        f"Percentage without higher education that earn >50K: {results['lower_education_rich']:.1f}%"
    )
    print(f"Min work time: {results['min_work_hours']} hours/week")
    print(
        f"Percentage of rich among those who work fewest hours: {results['rich_percentage']:.1f}%"
    )
    print("Country with highest percentage of rich:",
          results['highest_earning_country'])
    print(
        f"Highest percentage of rich people in country: {results['highest_earning_country_percentage']:.1f}%"
    )
    print("Top occupations in India:", results['top_IN_occupation'])

  return results
