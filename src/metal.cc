#include "src/bindings.h"
#include "mlx/backend/gpu/device_info.h"

namespace metal_ops {

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
DeviceInfo() {
  return mx::gpu::device_info(0);
}

}  // namespace metal_ops

void InitMetal(napi_env env, napi_value exports) {
  napi_value metal = ki::CreateObject(env);
  ki::Set(env, exports, "metal", metal);

  ki::Set(env, metal,
          "isAvailable", &mx::metal::is_available,
          "startCapture", &mx::metal::start_capture,
          "stopCapture", &mx::metal::stop_capture,
          "deviceInfo", &metal_ops::DeviceInfo);
}
