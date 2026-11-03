Results of the teammates' code (github.com/HoyngM/group-11-pm, main = 6a45964), run on 3 Nov 2026
with Python 3.12 and the packages of Project/code/requirements.txt:

  python main.py --datasets BPIC11_f1 --strategies apriori impressed   (same for BPIC11_f2, BPIC11_f3)
  python main.py --datasets synthetic shuffled_within_case shuffled_outcomes --strategies apriori impressed
  python main.py --datasets SIM_rare_pair SIM_frequent_pair --strategies apriori impressed
  python main.py --datasets BPIC11_f1 --strategies apriori apriori_support0.2_top25   (E4, same for f2, f3)

summary.csv merges the summary.csv files of these runs (E4 rows: strategy apriori_support0.2_top25).
A single "python main.py" writes the same files into one results/ folder.
