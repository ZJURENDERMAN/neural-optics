#include "scene_parser.hpp"

int main() {
    // 方式 1: 直接加载
    auto scene = diff_optics::load_scene("scene.json");
    scene->print();
    
    // 方式 2: 使用 parser 对象
    diff_optics::SceneParser parser;
    parser.set_base_dir("experiments/");
    auto scene2 = parser.load("another_scene.json");
    
    // 方式 3: 从字符串解析
    std::string json_str = R"({
        "spectrums": [{"name": "test", "type": "constant", "value": 1.0}],
        "surfaces": [{"name": "plane", "type": "rectangle_plane", "size": [10, 10]}]
    })";
    auto scene3 = diff_optics::parse_scene_string(json_str);
    
    return 0;
}