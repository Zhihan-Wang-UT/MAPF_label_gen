#include <cstdio>
#include <fstream>
#include <iostream>
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
#include <algorithm>    // std::min
#include <cmath>        // std::abs
#include <stdlib.h>

#include <torch/torch.h>
#include <torch/script.h>



using namespace std;

static int MY_INF = 999999;
static int STEP = 10;
// static string MAP_LOC = "../datasets/map/";
string MAP_LOC = "./";
bool PRINT_RESULT = 1;

template <class T>
using Map2d = vector<vector<T>>;

void ToFile2d(const vector<vector<int>>& vec2, const string& tensorFile);
bool file2lines(const string& fname, vector<string>& lines);

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
    vector<vector<Map2d<bool>>> kSubPath; // k * timeunit * Map2d< True iff cell is visitable at timeunit t on k-sub Path>
    vector<Map2d<int>> OnePath; // timeunit * Map2d
    int opt_path_length;

    string toString() const;
};

string Agent::toString() const {
    stringstream ss;
    ss << "<" << startpos.x << "," << startpos.y << ">";
    ss << " -> ";
    ss << "<" << goalpos.x << "," << goalpos.y << ">";
    ss << " ";
    ss << "Longest Path:" << kSubPath[0].size();
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
    string _scenfile;
    string _mapfile;
    string _output_prefix;
    string _output_dir;

    int _height;
    int _width;
    int _kmax;
    int _num_agents;
    int _num_agents_cap;

    Map2d<bool> _reachableMap;
    Map2d<bool> _obstacleMap;
    vector<Map2d<int>> _collisionMap;
    vector<Map2d<int>> _singlePathCollisionMap;
    vector<int> _n_to_runs;

    vector<Agent> _agents;

    string lines2agents(const vector<string>& lines);
    void lines2map2d(const vector<string>& lines);
    Map2d<int> GetDistFromPosMap(Position);
    void PeekAgentPathLength(Agent& agent);
    void SolveAgent(Agent& agent);
    std::vector<Position> GetAvailMove(Position);

    void Load(const string& sceneFile);
    void Solve();
    void ToFile(const string& tensorFile="tensor.pt"); // Archived


    bool isCloserToEdnge(Position A, Position B);
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
bool file2lines(const string& fname, vector<string>& lines) {
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
    _mapfile = MAP_LOC + lines2agents(scene_lines);

    vector<string> map_lines;
    if (file2lines(_mapfile, map_lines) == false) {
        printf("error reading file %s", _mapfile.data());
        exit(-1);
    }
    lines2map2d(map_lines);

    for (int n = STEP; n <= min(_agents.size(), static_cast<size_t>(_num_agents_cap)); n+=STEP){
        _n_to_runs.push_back(n);
    }

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

bool Datagen::isCloserToEdnge(Position A, Position B){
    int ax = static_cast<int>(A.x);
    int ay = static_cast<int>(A.y);
    int bx = static_cast<int>(B.x); 
    int by = static_cast<int>(B.y);
    int a_dist_to_edge = min(min( _width - ax, ax  ), min( _height - ay, ay ) );
    int b_dist_to_edge = min(min( _width - bx, bx  ), min( _height - by, by ) );
    return a_dist_to_edge < b_dist_to_edge;
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

    struct doComparePos{
        doComparePos( const Datagen& datagen ):
        _inner_width(datagen._width),
        _inner_height(datagen._height)
        {
        }
        const int _inner_width;
        const int _inner_height;

        bool operator()(Position A, Position B){
            int ax = static_cast<int>(A.x);
            int ay = static_cast<int>(A.y);
            int bx = static_cast<int>(B.x); 
            int by = static_cast<int>(B.y);
            int a_dist_to_edge = min(min( _inner_width - ax, ax  ), min( _inner_height - ay, ay ) );
            int b_dist_to_edge = min(min( _inner_width - bx, bx  ), min( _inner_height - by, by ) );
            return a_dist_to_edge < b_dist_to_edge;
        }
    };

    // DDM: Make the agent path as far from center as possible (as close to one of the edges as possible)
    sort(res.begin(), res.end(), doComparePos(*this));
    return res;
}

void Datagen::SolveAgent(Agent& agent) {
    Map2d<int> distanceFromStartMap = GetDistFromPosMap(agent.startpos);

    Map2d<int> distanceFromGoalMap = GetDistFromPosMap(agent.goalpos);
    // build up agent.kSubPath


    //TODO: remove
    agent.kSubPath.resize(_kmax,
        vector<vector<vector<bool>>>(agent.opt_path_length + _kmax + 1, 
        vector<vector<bool>>(_height, 
        vector<bool>(_width, 0))));

    
    // Iterate over cells
    for (size_t i = 0; i < _height; ++i) {
        for (size_t j = 0; j < _width; ++j) {
            
            // Iterate over k
            for (int k = 0; k < _kmax; ++k){
                int distanceFromStart = distanceFromStartMap[i][j];
                int distanceFromGoal = distanceFromGoalMap[i][j];

                // the opt path among all path that visit <i,j> will arrivie <i,j> at distanceFromStart, 
                // if there are excess time, the agent can choose to stay at <i,j> for a few moves
                for (int stay = 0; stay <= agent.opt_path_length + k - distanceFromStart - distanceFromGoal; ++stay){
                    agent.kSubPath[k][distanceFromStart + stay][i][j] = true;
                }
            }

        }
    }
    

    agent.OnePath.resize(agent.opt_path_length + 1, vector<vector<int>>(_height, vector<int>(_width, 0)));
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

}

bool myFlip(bool a, bool b) { return !b; }

template <typename T>
int myAdd(int a, T b) {
    return a + b;
}

int myMinusOneOnPositive(int a, int b) { return max(0, a - 1); }

// bool check_

string GetOutputFileName(string output_dir, string output_prefix, int num_agents, string feature){
    string command = "mkdir -p " + output_dir + "/" + to_string(num_agents);
    system(command.c_str()); // bad practice;
    string res = output_dir + "/" + to_string(num_agents) + "/" + output_prefix + "_" 
        + to_string(num_agents) + "_"  
        + feature 
        + ".pt";
    // cout << res << endl;
    return res;
}

void Datagen::Solve() {
    cout << "in Solve" << endl;

    Map2d<bool> obs_map(_height, vector<bool>(_width, 0));
    ReduceMap(obs_map, _reachableMap, &myFlip);
    _obstacleMap = obs_map;

    for (int n : _n_to_runs){
        vector<Agent>first_n_agents(_agents.begin(), _agents.begin()+n);
        Map2d<bool> first_n_agents_startLocationMap = MarkPos(first_n_agents, 1);
        Map2d<bool> first_n_agents_goalLocationMap = MarkPos(first_n_agents, 0);

        // convert and store in Map2d<int>
        Map2d<int> startmap(_height, vector<int>(_width, 0));
        Map2d<int> goalmap(_height, vector<int>(_width, 0));
        Map2d<int> obstaclemap(_height, vector<int>(_width, 0));
        ReduceMap(startmap, first_n_agents_startLocationMap, &myAdd);
        ReduceMap(goalmap, first_n_agents_goalLocationMap, &myAdd);
        ReduceMap(obstaclemap, _obstacleMap, &myAdd);
        


        ToFile2d(startmap, GetOutputFileName(_output_dir, _output_prefix, n, "startLoc"));
        ToFile2d(goalmap, GetOutputFileName(_output_dir, _output_prefix, n, "goalLoc"));
        ToFile2d(obstaclemap, GetOutputFileName(_output_dir, _output_prefix, n, "obstacle"));
    }



    size_t tmp = _agents.size();

    _agents.resize(_n_to_runs.back());

    // Store the max kSubPath Length among all agents.
    size_t max_kSubPath_size = 0;
    for (auto& ag : _agents) {
        PeekAgentPathLength(ag);
        max_kSubPath_size = max(max_kSubPath_size, (size_t)ag.opt_path_length + _kmax);
        // printf("<%lu,%lu> -> <%lu,%lu> opt = %d\n",ag.startpos.x,ag.startpos.y,ag.goalpos.x,ag.goalpos.y,ag.opt_path_length);
    }


    printf("_agents.size() = %lu -> %lu ; max_kSubPath_size = %lu\n", tmp, _agents.size(), max_kSubPath_size);
     
    /* ######### TODO ############*/
    // Start working on kSubPath collision
    // _collisionMap.resize(_height, vector<int>(_width, 0));

    vector<vector<Map2d<int>>> kTimeCollision(_kmax + 1, 
        vector<vector<vector<int>>>(max_kSubPath_size, 
        vector<vector<int>>(_height, 
        vector<int>(_width, 0))));
    

    Map2d<int> singlePathCollision(_height, vector<int>(_width, 0));

    
    
    // If an agent visit some cell on (k)SubPath at time = i , mark the cell on timeCollision[k][i]
    for (int i = 0; i < static_cast<int>(_agents.size()); i+=STEP){
        for (int j = 0; j < STEP; ++j){
            Agent ag = _agents[i+j];
        
            SolveAgent(ag);

            //work out kSubPath data and stores in kTimeCollision
            for (int k = 0; k < _kmax; ++k){
                while (ag.kSubPath[k].size() < max_kSubPath_size){
                    ag.kSubPath[k].push_back(ag.kSubPath[k].back());
                }
                
                for (int t = 0; t < static_cast<int>(max_kSubPath_size); ++t){
                    ReduceMap(kTimeCollision[k][t], ag.kSubPath[k][t], &myAdd);
                }


            }
            ag.kSubPath.clear();
            ag.kSubPath.shrink_to_fit();

            //work out SinglePath data and stores in singlePathCollision
            for (int t = 0; t < static_cast<int>(ag.opt_path_length+1); ++t){
                ReduceMap(singlePathCollision, ag.OnePath[t], &myAdd);
            }
            ag.OnePath.clear();
            ag.OnePath.shrink_to_fit();
        }

        // save kSupPath Collision to k seperate tensor files
        vector<vector<Map2d<int>>> kTimeCollision_tmp(kTimeCollision);
        for (int k = 0; k < _kmax; ++k){
            Map2d<int> kCollision(_height, vector<int>(_width, 0));
            for (int t = 0; t < static_cast<int>(max_kSubPath_size); ++t){
                ReduceMap(kTimeCollision_tmp[k][t], kTimeCollision_tmp[k][t], &myMinusOneOnPositive);
                ReduceMap(kCollision, kTimeCollision_tmp[k][t], &myAdd);
            }
            ToFile2d(kCollision, GetOutputFileName(_output_dir, _output_prefix, i+10, to_string(k) + "subPathColl"));
        }

        // save SinglePathCollision to a file
        ToFile2d(singlePathCollision, GetOutputFileName(_output_dir, _output_prefix, i+10, "singlePathHeat"));

    }

    printf("done with _collisionMap\n");

    fflush(stdout);
}

template <typename T>
vector<T> linearize2d(const vector<vector<T>>& vec2) {
    vector<T> vec;
    for (const auto& v : vec2) {
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


// convert a 2d vector to tensor
void ToFile2d(const vector<vector<int>>& vec2, const string& tensorFile){
    auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided);
    int height = static_cast<int>(vec2.size());
    int width = static_cast<int>(vec2[0].size());
    auto tensor2d = torch::empty(height * width, options);
    int* data = tensor2d.data_ptr<int>();
    for (const auto& i : vec2) {
        for (const auto& j : i) {
            *data++ = j;
        }
    }

    auto tensor2d_bytes = torch::jit::pickle_save(tensor2d.resize_({height,width}));
    std::ofstream fout(tensorFile, std::ios::out | std::ios::binary);
    fout.write(tensor2d_bytes.data(), tensor2d_bytes.size());
    fout.close();
}

void Datagen::ToFile(const string& tensorFile) {
    cout << "Datagen::ToFile(const string& tensorFile) -> Commented out" << endl; 
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        printf("need argc == %d, got %d", 6, argc);
        printf("useage: ./datagen <scene_file> <map_dir> <output_dir> <output_file_prefix> <num_agents_cap>\n");
        exit(-1);
    }

    Datagen datagen;

    MAP_LOC = argv[2];
    datagen._scenfile = argv[1];
    datagen._output_dir = argv[3];
    datagen._output_prefix = argv[4];
    datagen._num_agents_cap = stoi(argv[5]);
    cout << "Load()" << endl;

    datagen.Load(datagen._scenfile);
    datagen._kmax = 3;
    cout << "Solve()" << endl;
    datagen.Solve();
}
