#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <numeric>

#include <mlx/mlx.h>
#include <kizunapi.h>

namespace mx = mlx::core;

// Teach kizunapi how to serialize/deserialize SmallVector<T> (used for Shape
// and other types in MLX >= 0.26). Mirrors the std::vector<T> specialization.
namespace ki {

template<typename T, unsigned N, typename A>
struct Type<mlx::core::SmallVector<T, N, A>> {
  static constexpr const char* name = "Array";
  static napi_status ToNode(napi_env env,
                            const mlx::core::SmallVector<T, N, A>& vec,
                            napi_value* result) {
    napi_status s = napi_create_array_with_length(env, vec.size(), result);
    if (s != napi_ok) return s;
    for (size_t i = 0; i < vec.size(); ++i) {
      napi_value el;
      s = ConvertToNode(env, vec[i], &el);
      if (s != napi_ok) return s;
      s = napi_set_element(env, *result, i, el);
      if (s != napi_ok) return s;
    }
    return napi_ok;
  }
  static std::optional<mlx::core::SmallVector<T, N, A>> FromNode(
      napi_env env, napi_value value) {
    // Read as std::vector then convert to SmallVector.
    auto vec = Type<std::vector<T>>::FromNode(env, value);
    if (!vec) return std::nullopt;
    return mlx::core::SmallVector<T, N, A>(vec->begin(), vec->end());
  }
};

}  // namespace ki

using OptionalAxes = std::variant<std::monostate, int, std::vector<int>>;
using ScalarOrArray = std::variant<bool, float, mx::array>;

// Read args into a container of types (vector or SmallVector).
template<typename Container>
bool ReadArgs(ki::Arguments* args, Container* results) {
  using T = typename Container::value_type;
  while (args->RemainingsLength() > 0) {
    std::optional<T> a = args->GetNext<T>();
    if (!a) {
      args->ThrowError(ki::Type<T>::name);
      return false;
    }
    results->push_back(std::move(*a));
  }
  return true;
}

// Convert the type to string.
template<typename T>
std::string ToString(napi_value value, napi_env env) {
  std::optional<T*> self = ki::FromNodeTo<T*>(env, value);
  if (!self)
    return std::string("The object has been destroyed.");
  std::ostringstream ss;
  ss << *self.value();
  return ss.str();
}

// Define the toString method for type's prototype.
template<typename T>
void DefineToString(napi_env env, napi_value prototype) {
  auto symbol = ki::SymbolFor("nodejs.util.inspect.custom");
  ki::Set(env, prototype,
          "toString", ki::MemberFunction(&ToString<T>),
          symbol, ki::MemberFunction(&ToString<T>));
}

// If input is one int, put it into a Shape, otherwise just return the Shape.
mx::Shape PutIntoShape(std::variant<int, mx::Shape> shape);

// If input is one int, put it into a vector, otherwise just return the vector.
inline std::vector<int> PutIntoVector(std::variant<int, std::vector<int>> v) {
  if (auto i = std::get_if<int>(&v); i)
    return {*i};
  return std::move(std::get<std::vector<int>>(v));
}

// Get axis arg from js value.
std::vector<int> GetReduceAxes(OptionalAxes value, int dims);

// Convert a ScalarOrArray arg to array.
mx::array ToArray(ScalarOrArray value,
                  std::optional<mx::Dtype> dtype = std::nullopt);

// Execute the function and wait it to finish.
napi_value AwaitFunction(
    napi_env env,
    std::function<napi_value()> func,
    std::function<napi_value(napi_env, napi_value)> cpp_then,
    std::function<void(napi_env)> cpp_finally);

#endif  // SRC_UTILS_H_
