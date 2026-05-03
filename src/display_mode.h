#pragma once

enum class DisplayMode {
    Solid     = 0,
    Energy    = 1,
    Moisture  = 2,
    Wireframe = 3
};

inline const char* displayModeName(DisplayMode mode)
{
    switch (mode) {
        case DisplayMode::Solid:     return "Solid";
        case DisplayMode::Energy:    return "Energy";
        case DisplayMode::Moisture:  return "Moisture";
        case DisplayMode::Wireframe: return "Wireframe";
    }
    return "Unknown";
}
