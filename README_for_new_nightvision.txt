Build:
g++ -std=c++17 night_vision.cpp -o night_vision `pkg-config --cflags --libs opencv4` -lpthread

Run examples:
1. Normal (enhance + stream): ./night_vision
2. Record raw dataset for 20s (default): ./night_vision --record scene1.avi
3. Record for 60s: ./night_vision --record scene1.avi --seconds 60
4. Record AND also stream enhanced preview while recording:
    ./night_vision --record scene1.avi --seconds 30 --stream-enhanced