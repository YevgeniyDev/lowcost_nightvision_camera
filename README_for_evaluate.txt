Build:
g++ -std=c++17 evaluate_baselines.cpp -o evaluate_baselines \ `pkg-config --cflags --libs opencv4`
or (if errors)
g++ -std=c++17 evaluate_baselines.cpp -o evaluate_baselines $(pkg-config --cflags --libs opencv4)


Run:
./evaluate_baselines scene1.avi scene1_results.csv 1
./evaluate_baselines scene2.avi scene2_results.csv 1

You’ll get:
sceneX_results.csv (numbers for your table)
example_f*_Method.png (qualitative panels)