#pragma once
#include "IR/MLIRExtension.h"

using namespace mlir;

namespace KernelCodegen {

struct Expression;

struct Operand {
  enum class OperandType {
    Value = 0,
    Integer = 1,
    Expression = 2,
  };
  Operand() = default;
  explicit Operand(const Value& val_) {
    type = OperandType::Value;
    value = val_;
  }
  explicit Operand(int integer_) {
    type = OperandType::Integer;
    integer = integer_;
  }
  explicit Operand(std::shared_ptr<Expression> expr_) {
    type = OperandType::Expression;
    expr = expr_;
  }
  OperandType type;
  Value value;
  int integer;
  std::shared_ptr<Expression> expr;
};

enum class Operator {
  Add = 0,
  Mul = 1,
  Constant,
};

struct Expression {
  Expression(Operator op_, Operand& left_, Operand& right_) : 
    op(op_), left(left_), right(right_) {}
  Expression(int integer) {
    op = Operator::Constant;
    left = Operand(integer);
  }
  Operator op;
  Operand left;
  Operand right;
};

std::shared_ptr<Expression> Add(const Value& val, int integer);

std::shared_ptr<Expression> Add(std::shared_ptr<Expression> left_expr, int integer);

std::shared_ptr<Expression> Mul(const Value& val, int integer);

std::shared_ptr<Expression> Mul(std::shared_ptr<Expression> left_expr, int integer);

std::shared_ptr<Expression> Constant(int integer);

}