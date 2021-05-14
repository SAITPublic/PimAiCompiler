#include "common/file_parser.hpp"

std::optional<std::string> parseFile(const std::string& config_filename) {
    std::ifstream file(config_filename.data());
    if (file) {
        std::ostringstream file_string;
        file_string << file.rdbuf();
        return file_string.str();
    }
    return {};
}
