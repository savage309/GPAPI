#pragma once

//! Common defines used to set the project configuration

/// Sets the current log level. LogType::Info, LogType::Warning, LogType::Error and LogType::None are available
/// When Info is set, log messages with types Info, Warning and Error are printed
/// When Warning is set, log messages with types Warning and Error are printed
/// When Error is set, only log messages with type Error are printed
/// When None is set, nothing is printed
#define LOG_LEVEL LogTypeInfo

/// Sets the target, for which the project should be build. TARGET_CUDA, TARGET_OPENCL and TARGET_NATIVE are available
#define TARGET_CUDA
//#define TARGET_OPENCL
//#define TARGET_NATIVE
