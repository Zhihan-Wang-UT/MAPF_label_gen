#include <cstdio>
#include <fstream>
#include <iostream>
// #include "CBS.h"
#include <chrono>
#include <iterator>
#include <ostream>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <vector>

#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>


using namespace std;

static int MY_INF = 999999;
// static string MAP_LOC = "../datasets/map/";
string MAP_LOC = "./";
bool PRINT_RESULT = 0;

template <class T>
using Map2d = vector<vector<T>>;
// using Position = pair<size_t, size_t>;

struct Position {
    size_t x;
    size_t y;
    Position(size_t _x, size_t _y) {
        x = _x;
        y = _y;
    }
    Position(){
        x = 999;
        y = 999;
    }
};

struct Agent {
    Position startpos;
    Position goalpos;
    vector<Map2d<bool>> kSubPath;
    vector<Map2d<int>> OnePath;
    int opt_path_length;

    string toString() const;
};

string Agent::toString() const {
    stringstream ss;
    ss << "<" << startpos.x << "," << startpos.y << ">";
    ss << " -> ";
    ss << "<" << goalpos.x << "," << goalpos.y << ">";
    ss << " ";
    ss << "Longest Path:" << kSubPath.size();
    return ss.str();
}

template <typename T>
string Map2string(Map2d<T> m) {
    stringstream ss;
    for (const auto& l : m) {
        for (const auto& x : l) {
            ss << int(x) << " ";
        }
        ss << "\n";
    }
    return ss.str();
}

class Datagen {
   public:
    string _mapfile;

    int _height;
    int _width;
    int _k;
    int _num_agents;

    Map2d<bool> _startLocationMap;
    Map2d<bool> _goalLocationMap;
    Map2d<bool> _reachableMap;
    Map2d<bool> _obstacleMap;
    Map2d<int> _collisionMap;
    Map2d<int> _singlePathCollisionMap;

    vector<Agent> _agents;

    bool file2lines(const string& fname, vector<string>& lines);

    string lines2agents(const vector<string>& lines);
    void lines2map2d(const vector<string>& lines);
    Map2d<int> GetDistFromPosMap(Position);
    void PeekAgentPathLength(Agent& agent);
    void SolveAgent(Agent& agent);
    std::vector<Position> GetAvailMove(Position);

    void Load(const string& sceneFile);
    void Solve();
    void ToFile(const string& tensorFile="tensor.pt");
    Map2d<bool> MarkPos(vector<Agent>& agents, bool startpos = 1);
};

template <typename T, typename U>
void ReduceMap(Map2d<T>& accumulated, Map2d<U>& increment, T (*func)(T, U)) {
    for (size_t i = 0; i < accumulated.size(); ++i) {
        for (size_t j = 0; j < accumulated[0].size(); ++j) {
            accumulated[i][j] = func(accumulated[i][j], increment[i][j]);
        }
    }
}

// return false on error
bool Datagen::file2lines(const string& fname, vector<string>& lines) {
    ifstream myfile(fname.c_str(), ios_base::in);
    string line;
    if (myfile.is_open()) {
        for (; getline(myfile, line);) {
            lines.push_back(line);
        }
        myfile.close();
        return true;
    }
    return false;
}

void Datagen::lines2map2d(const vector<string>& lines) {
    string dimension_patten = "^[a-zA-Z]+ (\\d+)$";

    smatch match;
    auto matcher = regex(dimension_patten);

    regex_search(lines[1], match, matcher);
    _height = stoi(match.str(1));
    regex_search(lines[2], match, matcher);
    _width = stoi(match.str(1));

    _reachableMap.resize(_height, vector<bool>(_width, 0));
    // drop firstline;
    for (size_t i = 4; i < lines.size(); ++i) {
        if (lines[i].size() != _width) {
            printf("inconsistancy, expecting _width = %d, got %lu", _width, lines[i].size());
        }
        for (size_t j = 0; j < lines[i].size(); ++j) {
            _reachableMap[i - 4][j] = lines[i][j] == '.';
        }
    }

    // _reachableMap = map;
}

string Datagen::lines2agents(const vector<string>& lines) {
    smatch match;
    string mapline_patten = "^\\d+\\s(\\S+)(\\s\\d+){2}\\s(\\d+)\\s(\\d+)\\s(\\d+)\\s(\\d+)(\\s\\S+)$";
    auto matcher = regex(mapline_patten);

    regex_search(lines[1], match, matcher);
    _mapfile = match.str(1);

    for (size_t i = 1; i < lines.size(); ++i) {
        regex_search(lines[i], match, matcher);
        Position startpos = Position(stoi(match.str(4)), stoi(match.str(3)));
        Position goalpos = Position(stoi(match.str(6)), stoi(match.str(5)));
        Agent a{startpos, goalpos};
        _agents.push_back(a);
    }

    return _mapfile;
}

Map2d<bool> Datagen::MarkPos(vector<Agent>& agents, bool startpos) {
    Map2d<bool> map(_height, vector<bool>(_width, 0));
    for (Agent& a : agents) {
        Position pos = startpos ? a.startpos : a.goalpos;
        map[pos.x][pos.y] = true;
    }
    return map;
}

void Datagen::Load(const string& scenefile) {
    vector<string> scene_lines;
    if (file2lines(scenefile, scene_lines) == false) {
        printf("error reading file %s", scenefile.data());
        exit(-1);
    }
    string mapfile = MAP_LOC + lines2agents(scene_lines);

    vector<string> map_lines;
    if (file2lines(mapfile, map_lines) == false) {
        printf("error reading file %s", mapfile.data());
        exit(-1);
    }
    lines2map2d(map_lines);

    // cout << Map2string(_reachableMap);
}

Map2d<int> Datagen::GetDistFromPosMap(Position initPos) {
    Map2d<int> distanceFromStartMap(_height, vector<int>(_width, MY_INF));
    queue<pair<Position, int>> toVisit;
    toVisit.push(make_pair(initPos, 0));
    while (!toVisit.empty()) {
        Position pos = toVisit.front().first;
        int distance = toVisit.front().second;
        toVisit.pop();

        if (distance >= distanceFromStartMap[pos.x][pos.y]) {
            continue;
        }
        distanceFromStartMap[pos.x][pos.y] = distance;
        distance += 1;
        if (pos.x > 0 && _reachableMap[pos.x - 1][pos.y]) {
            toVisit.push(make_pair(Position(pos.x - 1, pos.y), distance));
        }
        if (pos.y > 0 && _reachableMap[pos.x][pos.y - 1]) {
            toVisit.push(make_pair(Position(pos.x, pos.y - 1), distance));
        }
        if (pos.x + 1 < _height && _reachableMap[pos.x + 1][pos.y]) {
            toVisit.push(make_pair(Position(pos.x + 1, pos.y), distance));
        }
        if (pos.y + 1 < _width && _reachableMap[pos.x][pos.y + 1]) {
            toVisit.push(make_pair(Position(pos.x, pos.y + 1), distance));
        }
    }
    return distanceFromStartMap;
}

bool once = true;
void Datagen::PeekAgentPathLength(Agent& agent) {
    Map2d<int> distanceFromStartMap = GetDistFromPosMap(agent.startpos);
    agent.opt_path_length = distanceFromStartMap[agent.goalpos.x][agent.goalpos.y];
    if (once && agent.opt_path_length == MY_INF){
        printf("problem unsolvable for some agents %s", agent.toString().c_str());
        exit(-1);
        cout << Map2string(distanceFromStartMap) << endl;
        once = false;
    }
}

std::vector<Position> Datagen::GetAvailMove(Position pos){
    std::vector<Position> res;
    int x = static_cast<int>(pos.x);
    int y = static_cast<int>(pos.y);
    if (x + 1 < _height){
        res.push_back(Position(x+1,y));
    }
    if (x >= 1 ){
        res.push_back(Position(x-1,y));
    }
    if (y + 1 < _width){
        res.push_back(Position(x,y+1));
    }
    if (y >= 1 ){
        res.push_back(Position(x,y-1));
    }
    res.push_back(Position(x,y));
    if (res.size() == 1){
        printf("No Avail Move, that is IMPOSSIBLE!\n");
    }
    return res;
}

void Datagen::SolveAgent(Agent& agent) {
    Map2d<int> distanceFromStartMap = GetDistFromPosMap(agent.startpos);

    Map2d<int> distanceFromGoalMap = GetDistFromPosMap(agent.goalpos);

    // build up agent.kSubPath
    agent.kSubPath.resize(agent.opt_path_length + _k + 1, vector<vector<bool>>(_height, vector<bool>(_width, 0)));

    for (int stay = 0; stay <= _k; ++stay) {
        for (size_t i = 0; i < _height; ++i) {
            for (size_t j = 0; j < _width; ++j) {
                int distanceFromStart = distanceFromStartMap[i][j];
                int distanceFromGoal = distanceFromGoalMap[i][j];
                if (distanceFromStart + distanceFromStart <= agent.opt_path_length + _k) {
                    agent.kSubPath[distanceFromStart + stay][i][j] = true;
                }
                // printf("%d",agent.opt_path_length - pathViaPosLength+k);
                // printf("(stay=%lu) distance from start %d, distance from goal %d, globalSortestPathLength %d\n",stay,distanceFromStart,distanceFromGoal,opt_path_length);
            }
        }
    }

    agent.OnePath.resize(agent.opt_path_length + _k + 1, vector<vector<int>>(_height, vector<int>(_width, 0)));
    Position pos = agent.startpos;
    for (int t = 0; t <= agent.opt_path_length; ++t){
        // printf("<%lu,%lu> -> <%lu,%lu> @ <%lu,%lu>\n",agent.startpos.x, agent.startpos.y,agent.goalpos.x, agent.goalpos.y, pos.x, pos.y);
        agent.OnePath[t][pos.x][pos.y] = 1;
        if (pos.x == agent.goalpos.x && pos.y == agent.goalpos.y){
            break;
        }
        bool updated = false;
        for (auto& next_pos : GetAvailMove(pos)){
            // printf("\t t = %d <%lu,%lu> from start:%d, from end:%d\n",t, next_pos.x, next_pos.y, distanceFromStartMap[next_pos.x][next_pos.y], distanceFromGoalMap[next_pos.x][next_pos.y]);
            if (distanceFromStartMap[next_pos.x][next_pos.y] == t+1 && distanceFromGoalMap[next_pos.x][next_pos.y] == agent.opt_path_length-t-1){
                pos = next_pos;
                updated = true;
                break;
            }
        }
        if (!updated){
            printf("Look here! 307, something went wrong, not next_pos is selected \n");
            break;
        }
    }


    // for (Map2d<bool> m : agent.kSubPath){
    //     cout << Map2string(m) << endl;
    // }
}

bool myFlip(bool a, bool b) { return !b; }

template <typename T>
int myAdd(int a, T b) {
    return a + b;
}

int myMinusOneOnPositive(int a, int b) { return max(0, a - 1); }

// bool check_

void Datagen::Solve() {
    cout << "in Solve" << endl;

    if (_agents.size() < _num_agents){
        printf("Error, scene file does not contain enough agents: got: %lu need: %d\n", _agents.size(), _num_agents);
    } else{
        _agents.resize(_num_agents);
    }

    _startLocationMap = MarkPos(_agents, 1);
    _goalLocationMap = MarkPos(_agents, 0);

    Map2d<bool> map(_height, vector<bool>(_width, 0));
    ReduceMap(map, _reachableMap, &myFlip);
    _obstacleMap = map;

    // Store the max kSubPath Length among all agents.
    size_t max_kSubPath_size = 0;
    for (auto& ag : _agents) {
        PeekAgentPathLength(ag);
        max_kSubPath_size = max(max_kSubPath_size, (size_t)ag.opt_path_length+_k);
        // printf("<%lu,%lu> -> <%lu,%lu> opt = %d\n",ag.startpos.x,ag.startpos.y,ag.goalpos.x,ag.goalpos.y,ag.opt_path_length);
    }

    printf("_agents.size() = %lu ; max_kSubPath_size = %lu\n" ,_agents.size(),max_kSubPath_size);
    
    fflush(stdout);

    // Start working on kSubPath collision
    _collisionMap.resize(_height, vector<int>(_width, 0));
    vector<Map2d<int>> timeCollision(max_kSubPath_size, vector<vector<int>>(_height, vector<int>(_width, 0)));
    
    // If an agent visit some cell at time = i, add mark the cell on timeCollision[i]
    for (auto& ag : _agents) {
        SolveAgent(ag);
        // Agents stays at their goal location once arrived.
        while (ag.kSubPath.size() < max_kSubPath_size) {
            ag.kSubPath.push_back(ag.kSubPath.back());
        }
        // timeCollision[i] += ag.kSubPath[i]
        for (size_t i = 0; i < max_kSubPath_size; ++i) {
            ReduceMap(timeCollision[i], ag.kSubPath[i], &myAdd);
        }
        // Clean up memory
        ag.kSubPath.clear();
        ag.kSubPath.shrink_to_fit();
    }

    printf("done with _collisionMap\n");

    // _collisionMap = SUM( collision over time )
    for (size_t i = 0; i < max_kSubPath_size; ++i){
        ReduceMap(timeCollision[i], timeCollision[i], &myMinusOneOnPositive);
        ReduceMap(_collisionMap, timeCollision[i], &myAdd);
    }
    
    // start working on _singlePathCollisionMap (Like that in DDM)
    _singlePathCollisionMap.resize(_height, vector<int>(_width, 0));
    vector<Map2d<int>> timeSinglePathCollision(max_kSubPath_size, vector<vector<int>>(_height, vector<int>(_width, 0)));
    for (auto& ag : _agents) {
        for (size_t t = 0; t < ag.opt_path_length; ++t){
            ReduceMap(timeSinglePathCollision[t], ag.OnePath[t], &myAdd);
        }
        ag.OnePath.clear();
        ag.OnePath.shrink_to_fit();
    }
    for (size_t t = 0; t<max_kSubPath_size; ++t){
        ReduceMap(_singlePathCollisionMap, timeSinglePathCollision[t], &myAdd);
    }


    if (PRINT_RESULT){
        cout << "///////// _collisionMap //////////////\n";
        cout << Map2string(_collisionMap);
        cout << "////////// _singlePathCollisionMap /////////////\n";
        cout << Map2string(_singlePathCollisionMap);
    }

    fflush(stdout);
}

template <typename T>
vector<T> linearize2d(const vector<vector<T>>& vec_vec) {
    vector<T> vec;
    for (const auto& v : vec_vec) {
        for (auto d : v) {
            vec.push_back(d);
        }
    }
    return vec;
}

template <typename T>
vector<T> linearize3d(const vector<vector<vector<T>>>& v3) {
    vector<T> vec;
    for (const auto& v2 : v3) {
        for (const auto& v : v2) {
            for (const auto& item : v){
                vec.push_back(item);
            }
        }
    }
    return vec;
}


void Datagen::ToFile(const string& tensorFile) {
    Map2d<int> startLocationMap(_height, vector<int>(_width, 0));
    Map2d<int> goalLocationMap(_height, vector<int>(_width, 0));
    Map2d<int> obstacleMap(_height, vector<int>(_width, 0));
    Map2d<int> zerosMap(_height, vector<int>(_width, 0));
    Map2d<int> collisionMap(_height, vector<int>(_width, 0));
    Map2d<int> singlePathCollisionMap(_height, vector<int>(_width, 0));

    ReduceMap(startLocationMap, _startLocationMap, &myAdd);
    ReduceMap(goalLocationMap, _goalLocationMap, &myAdd);
    ReduceMap(obstacleMap, _obstacleMap, &myAdd);
    ReduceMap(collisionMap, _collisionMap, &myAdd);
    ReduceMap(singlePathCollisionMap, _singlePathCollisionMap, &myAdd);

    // vector<Map2d<int>> tmp = {zerosMap,zerosMap,zerosMap,zerosMap};
    vector<Map2d<int>> layers = {startLocationMap,goalLocationMap,obstacleMap,collisionMap,singlePathCollisionMap};

    auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided);

    auto blob3d = layers;
    auto tensor3d = torch::empty((int)blob3d.size() * _height * _width, options);
    int* data = tensor3d.data_ptr<int>();
    for (const auto& i : blob3d) {
        for (const auto& j : i) {
            for (const auto& k : j) {
                *data++ = k;
            }
        }
    }

    auto tensor3d_bytes = torch::jit::pickle_save(tensor3d.resize_({(int)layers.size(),_height,_width}));
    std::ofstream fout(tensorFile, std::ios::out | std::ios::binary);
    fout.write(tensor3d_bytes.data(), tensor3d_bytes.size());
    fout.close();

    // for (const auto& m : layers){
    //     cout << Map2string(m) << endl;
    //     torch::Tensor t = torch::from_blob(linearize2d(m).data(),{_height,_width},options);
    //     cout << t << endl << endl;;
    // }

    // auto bytes = torch::jit::pickle_save(torch::from_blob(linearize2d(collisionMap).data(),{_height,_width},options));
    // std::ofstream fout0("datagen2d.pt", std::ios::out | std::ios::binary);
    // fout0.write(bytes.data(), bytes.size());
    // fout0.close();

    // torch::Tensor tensor3d = torch::from_blob(linearize3d(layers).data(), {(int)layers.size(),_height,_width}, options);

    // // Do nothing for now;
    // Map2d<int> dummy(2,vector<int>(4,0));
    // int i = 1;
    // for (auto &r:dummy){
    //     for (auto & x : r){
    //         x = i++;
    //     }
    // }
    // vector<int> flat = linearize2d(dummy);
    // cout << Map2string(dummy) << endl;;

    // auto options = torch::TensorOptions().dtype(torch::kInt32);
    // torch::Tensor tharray = torch::from_blob(flat.data(), {2,4}, options);
    // std::cout << tharray << std::endl;
    // torch::Tensor tensor = torch::rand({2, 4});
    // std::cout << tensor << std::endl;


    // std::vector<torch::Tensor> tensor_vec = { tharray,tensor };
    // torch::save(tensor_vec, "my_tensor_vec.pt");
    // torch::save(tharray, "my_tensor.pt");


    // auto bytes = torch::jit::pickle_save(tharray);
    // std::ofstream fout("pickle_saved_tensor.pt", std::ios::out | std::ios::binary);
    // fout.write(bytes.data(), bytes.size());
    // fout.close();

    // bytes = torch::jit::pickle_save(tensor_vec);
    // std::ofstream fout1("pickle_saved_tensor_vec.pt", std::ios::out | std::ios::binary);
    // fout1.write(bytes.data(), bytes.size());
    // fout1.close();
    
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("need argc == %d, got %d", 5, argc);
        printf("useage: ./datagen <scene_file> <map_dir> <num_of_agents> <output_tensor_file>");
        exit(-1);
    }

    Datagen datagen;
    datagen._num_agents = stoi(argv[3]);

    MAP_LOC = argv[2];
    string scen_file = argv[1];
    cout << "Load()" << endl;
    datagen.Load(scen_file);

    datagen._k = 2;
    cout << "Solve()" << endl;
    datagen.Solve();

    string tensor_file = argv[4];
    cout << "ToFile()" << endl;
    datagen.ToFile(tensor_file);

    // Map2d<bool>
}
