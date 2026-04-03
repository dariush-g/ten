#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace ten::codegen {
	struct Expr {
		[[nodiscard]] virtual std::string emit_c() const = 0;

		virtual ~Expr() = default;
	};

	struct Stmt {
		[[nodiscard]] virtual std::string emit_c(int indent) const = 0;

		virtual ~Stmt() = default;
	};

	using ExprPtr = std::shared_ptr<Expr>;
	using StmtPtr = std::shared_ptr<Stmt>;

	struct Var : Expr {
		std::string name;

		explicit Var(std::string name) : name(std::move(name)) {
		}

		[[nodiscard]] std::string emit_c() const override { return name; }
	};

	struct IntLit : Expr {
		int value;

		explicit IntLit(const int value) : value(value) {
		}

		[[nodiscard]] std::string emit_c() const override { return std::to_string(value); }
	};

	struct FloatLit : Expr {
		float value;

		explicit FloatLit(const float value) : value(value) {
		}

		[[nodiscard]] std::string emit_c() const override {
			std::ostringstream ss;
			ss << value;
			std::string s = ss.str();
			if (s.find('.') == std::string::npos &&
			    s.find('e') == std::string::npos)
				s += ".0";
			return s + "f";
		}
	};

	struct BinOp : Expr {
		std::string op; // "+", "*", "+=", ">"
		ExprPtr lhs, rhs;

		explicit BinOp(std::string op, ExprPtr lhs, ExprPtr rhs)
			: op(std::move(op)), lhs(std::move(lhs)), rhs(std::move(rhs)) {
		}

		[[nodiscard]] std::string emit_c() const override {
			return lhs->emit_c() + " " + op + " " + rhs->emit_c();
		}
	};

	struct ArrayAccess : Expr {
		std::string array;
		ExprPtr index;

		ArrayAccess(std::string array, ExprPtr index)
			: array(std::move(array)), index(std::move(index)) {
		}

		[[nodiscard]] std::string emit_c() const override {
			return array + "[" + index->emit_c() + "]";
		}
	};

	struct Ternary : Expr {
		ExprPtr cond, then_expr, else_expr;

		Ternary(ExprPtr cond, ExprPtr then_expr, ExprPtr else_expr)
			: cond(std::move(cond)), then_expr(std::move(then_expr)),
			  else_expr(std::move(else_expr)) {
		}

		[[nodiscard]] std::string emit_c() const override {
			return cond->emit_c() + " ? " + then_expr->emit_c() + " : " +
			       else_expr->emit_c();
		}
	};

	struct Cast : Expr {
		std::string type;
		ExprPtr expr;

		Cast(std::string type, ExprPtr expr)
			: type(std::move(type)), expr(std::move(expr)) {
		}

		[[nodiscard]] std::string emit_c() const override {
			return "((" + type + ")" + expr->emit_c() + ")";
		}
	};

	static std::string indent_str(const int indent) {
		return std::string(static_cast<size_t>(indent * 4), ' ');
	}


	struct Assign : Stmt {
		ExprPtr lhs, rhs;
		std::string op; // "=" or "+="
		Assign(ExprPtr lhs, std::string op, ExprPtr rhs)
			: lhs(std::move(lhs)), rhs(std::move(rhs)), op(std::move(op)) {
		}


		[[nodiscard]] std::string emit_c(int indent) const {
			return indent_str(indent) + lhs->emit_c() + " " + op + " " +
			       rhs->emit_c() + ";\n";
		}
	};

	struct ForLoop : Stmt {
		std::string var;
		int start, end;
		std::vector<StmtPtr> body;

		ForLoop(std::string var, int start, int end)
			: var(std::move(var)), start(start), end(end) {
		}

		[[nodiscard]] std::string emit_c(const int indent) const {
			std::string s = indent_str(indent) + "for (int " + var + " = " +
			                std::to_string(start) + "; " + var + " < " +
			                std::to_string(end) + "; " + var + "++) {\n";
			for (auto &stmt: body)
				s += stmt->emit_c(indent + 1);
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

		[[nodiscard]] std::string emit_c(const int indent) const {
			return indent_str(indent) + type + "* " + name + " = (" + type +
			       "*)tensors[" + std::to_string(tensor_idx) + "];\n";
		}
	};

	struct Function : Stmt {
		std::string name;
		std::vector<StmtPtr> body;

		explicit Function(std::string name) : name(std::move(name)) {
		}

		[[nodiscard]] std::string emit_c(const int indent) const {
			std::string s = "void " + name + "(float** tensors, int n) {\n";
			for (auto &stmt: body)
				s += stmt->emit_c(indent + 1);
			s += "}\n";
			return s;
		}
	};
} // namespace ten::codegen
