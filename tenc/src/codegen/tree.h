#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ten::codegen {
struct Expr {
	virtual std::string emit() const = 0;
	virtual ~Expr() = default;
};

struct Stmt {
	virtual std::string emit(int indent) const = 0;
	virtual ~Stmt() = default;
};

using ExprPtr = std::shared_ptr<Expr>;
using StmtPtr = std::shared_ptr<Stmt>;

struct Var : Expr {
	std::string name;
	Var(std::string name) : name(std::move(name)) {}
	std::string emit() const override { return name; }
};

struct IntLit : Expr {
	int value;
	IntLit(int value) : value(value) {}
	std::string emit() const override { return std::to_string(value); }
};

struct BinOp : Expr {
	std::string op; // "+", "*", "+=", etc
	ExprPtr lhs, rhs;
	BinOp(std::string op, ExprPtr lhs, ExprPtr rhs)
		: op(std::move(op)), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
	std::string emit() const override {
		return lhs->emit() + " " + op + " " + rhs->emit();
	}
};

struct ArrayAccess : Expr {
	std::string array;
	ExprPtr index;
	ArrayAccess(std::string array, ExprPtr index)
		: array(std::move(array)), index(std::move(index)) {}
	std::string emit() const override {
		return array + "[" + index->emit() + "]";
	}
};

struct Ternary : Expr {
	ExprPtr cond, then_expr, else_expr;
	Ternary(ExprPtr cond, ExprPtr then_expr, ExprPtr else_expr)
		: cond(std::move(cond)), then_expr(std::move(then_expr)),
		  else_expr(std::move(else_expr)) {}
	std::string emit() const override {
		return cond->emit() + " ? " + then_expr->emit() + " : " +
			   else_expr->emit();
	}
};

struct Cast : Expr {
	std::string type;
	ExprPtr expr;
	Cast(std::string type, ExprPtr expr)
		: type(std::move(type)), expr(std::move(expr)) {}
	std::string emit() const override {
		return "((" + type + ")" + expr->emit() + ")";
	}
};

static std::string indent_str(int indent) {
	return std::string(indent * 4, ' ');
}

struct Assign : Stmt {
	ExprPtr lhs, rhs;
	std::string op; // "=" or "+="
	Assign(ExprPtr lhs, std::string op, ExprPtr rhs)
		: lhs(std::move(lhs)), op(std::move(op)), rhs(std::move(rhs)) {}
	std::string emit(int indent) const override {
		return indent_str(indent) + lhs->emit() + " " + op + " " + rhs->emit() +
			   ";\n";
	}
};

struct ForLoop : Stmt {
	std::string var;
	int start, end;
	std::vector<StmtPtr> body;
	ForLoop(std::string var, int start, int end)
		: var(std::move(var)), start(start), end(end) {}
	std::string emit(int indent) const override {
		std::string s = indent_str(indent) + "for (int " + var + " = " +
						std::to_string(start) + "; " + var + " < " +
						std::to_string(end) + "; " + var + "++) {\n";
		for (auto &stmt : body)
			s += stmt->emit(indent + 1);
		s += indent_str(indent) + "}\n";
		return s;
	}
};

struct DeclPtr : Stmt {
	std::string type, name;
	int tensor_idx;
	DeclPtr(std::string type, std::string name, int tensor_idx)
		: type(std::move(type)), name(std::move(name)), tensor_idx(tensor_idx) {
	}
	std::string emit(int indent) const override {
		return indent_str(indent) + type + "* " + name + " = (" + type +
			   "*)tensors[" + std::to_string(tensor_idx) + "];\n";
	}
};

struct Function : Stmt {
	std::string name;
	std::vector<StmtPtr> body;
	Function(std::string name) : name(std::move(name)) {}
	std::string emit(int indent) const override {
		std::string s = "void " + name + "(float** tensors, int n) {\n";
		for (auto &stmt : body)
			s += stmt->emit(indent + 1);
		s += "}\n";
		return s;
	}
};

} // namespace ten::codegen