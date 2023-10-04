#include "Expression.h"

// using namespace mlir;

namespace KernelCodegen {

std::shared_ptr<Expression> Add(const Value& val, int integer) {
  auto left = Operand(val);
  auto right = Operand(integer);
  return std::make_shared<Expression>(Operator::Add, left, right);
}

std::shared_ptr<Expression> Add(std::shared_ptr<Expression> left_expr, int integer) {
  auto left = Operand(left_expr);
  auto right = Operand(integer);
  return std::make_shared<Expression>(Operator::Add, left, right);
}

std::shared_ptr<Expression> Mul(const Value& val, int integer) {
  auto left = Operand(val);
  auto right = Operand(integer);
  return std::make_shared<Expression>(Operator::Mul, left, right);
}

std::shared_ptr<Expression> Mul(std::shared_ptr<Expression> left_expr, int integer) {
  auto left = Operand(left_expr);
  auto right = Operand(integer);
  return std::make_shared<Expression>(Operator::Mul, left, right);
}

std::shared_ptr<Expression> Constant(int integer) {
  return std::make_shared<Expression>(integer);
}

}