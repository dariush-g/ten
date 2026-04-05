#pragma once

enum CompileFlags : unsigned
{
    NONE = 0,
    EMIT_C = 1 << 0,
    DUMP_IR = 1 << 1,
    NO_OPT = 1 << 2,
    PRINT_C = 1 << 3,
    NO_TILE = 1 << 4,
    NO_FUSE = 1 << 5,
};
