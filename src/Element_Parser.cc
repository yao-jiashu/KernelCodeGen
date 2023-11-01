#include "Element_Parser.h"
#include "utils.h"

// used to sample static information

using namespace mlir;
using namespace KernelCodegen;

namespace KernelCodegen
{
	int Ele_Parser::search_var_idx(mlir::Value& variable, std::vector<std::pair<mlir::Value, std::string>>& variables)
	{
	int i = 0;
	for(; i<variables.size(); i++)
		if(variable == variables[i].first)
		break;
	if(i == variables.size())
		i = -1;
	return i;
	}

	std::string Ele_Parser::parse_variable_type(mlir::Type a_type, bool with_head, bool as_pointer)
	{
		std::string result = "";
		if (a_type.isa<mlir::MemRefType>())
		{
			// parse to an array pointer
			mlir::MemRefType real_type = a_type.dyn_cast<mlir::MemRefType>();
			uint32_t memspace_i = real_type.getMemorySpaceAsInt();
			if(with_head)
			{
				if(memspace_i == 1) 
					result = result + "__global__ ";	// global variable
				if(memspace_i == 2)
					result = result + "__shared__ ";	// shared variable
				if(memspace_i == 3)
					;	// register variable
			}
			mlir::Type ele_type = real_type.getElementType();
			if (ele_type.isF32())
				result = result + "float";

			if(as_pointer)
				result = result + "* ";
			else
				result = result + " ";
			return result;
		}
		if (a_type.isInteger(32))
		{
			result = result + "int32_t ";
			return result;
		}
		return result;
	}
	
	std::string Ele_Parser::parse_array_dim(mlir::MemRefType a_type)
	{
		std::string result = "";
		auto dims = a_type.getShape();
		for(int i=0; i<dims.size(); i++)
		{
			result = result + "[" + std::to_string(dims[i]) + "]";
		}
		return result;
	}

	void Ele_Parser::parse_gpuarth_dim(mlir::AffineForOp forOp, int gpu_arth_dim[][3])
	{
		auto hint_attr = forOp->getAttr("gpu.parallel_arch");
		assert(hint_attr != nullptr);
		auto hint_Str = hint_attr.dyn_cast<StringAttr>().str();
		auto lowerbound = forOp.getLowerBoundMap();
		auto upperbound = forOp.getUpperBoundMap();
		int step = forOp.getStep();
		assert(step == 1);
		int64_t lb = lowerbound.getSingleConstantResult();
		int64_t ub = upperbound.getSingleConstantResult();
		assert(lb == 0);
		if(hint_Str == "blockIdx.x")
			gpu_arth_dim[0][0] = ub;
		if(hint_Str == "blockIdx.y")
			gpu_arth_dim[0][1] = ub;
		if(hint_Str == "blockIdx.z")
			gpu_arth_dim[0][2] = ub;

		if(hint_Str == "threadIdx.x")
			gpu_arth_dim[1][0] = ub;
		if(hint_Str == "threadIdx.y")
			gpu_arth_dim[1][1] = ub;
		if(hint_Str == "threadIdx.z")
			gpu_arth_dim[1][2] = ub;

		return;
	}

	std::string Ele_Parser::parse_for_op_head(mlir::AffineForOp forOp, std::string for_idx_identifier)
	{
		std::string result = "for(int ";
		result = result + for_idx_identifier;
		auto lowerbound = forOp.getLowerBoundMap();
		auto upperbound = forOp.getUpperBoundMap();
		int step = forOp.getStep();
		int64_t lb = lowerbound.getSingleConstantResult();
		int64_t ub = upperbound.getSingleConstantResult();
		result = result + " = " + std::to_string(lb) + "; ";
		result = result + for_idx_identifier + " < " + std::to_string(ub) + "; ";
		result = result + for_idx_identifier + " += " + std::to_string(step) + ")";
		return result;
	}

	std::string Ele_Parser::parse_const_op(mlir::Operation* op_pointer, std::string var_identifier)
	{
		std::string type_string = "const ";
		std::string value_string;

		if(llvm::isa<mlir::arith::ConstantIndexOp>(*op_pointer))
		{
			type_string = type_string + "int ";
			auto concrete_op = dyn_cast<mlir::arith::ConstantIndexOp>(*op_pointer);
			value_string = std::to_string(concrete_op.value());
		}

		std::string result = type_string + var_identifier + " = " + value_string + ";";
		return result;
	}

	std::string Ele_Parser::parse_muli_op(mlir::Operation* op_pointer, std::string out, std::string in1, std::string in2, bool with_out_type)
	{
		std::string type_string;
		std::string value_string;

		if(llvm::isa<mlir::arith::ConstantIndexOp>(*op_pointer))
		{
			type_string = "int ";
			auto concrete_op = dyn_cast<mlir::arith::ConstantIndexOp>(*op_pointer);
			value_string = std::to_string(concrete_op.value());
		}

		std::string result = type_string + out + " = " + in1 + " * " + in2 + ";";
		return result;
	}

	std::string Ele_Parser::parse_vcopy_op(std::vector<mlir::Operation*> op_pointers, std::vector<std::pair<mlir::Value, std::string>>& variables_dict)
	{
		assert(op_pointers.size() == 3);

		mlir::memref::SubViewOp op_subview0 = dyn_cast<mlir::memref::SubViewOp>(*op_pointers[0]);
		mlir::memref::SubViewOp op_subview1 = dyn_cast<mlir::memref::SubViewOp>(*op_pointers[1]);
		mlir::memref::CopyOp op_copy = dyn_cast<mlir::memref::CopyOp>(*op_pointers[2]);

		bool flip = false;
		if(op_copy.getSource() == op_subview0.getResult())
			flip = false;
		else if(op_copy.getSource() == op_subview1.getResult())
			flip = true;
		else
		{
			printf("vec copy pattern not support error");
			assert(0 && "vec copy pattern not support error");
		}
			

		std::string result = "";
		std::string type_string;

		auto subview0_sizes = op_subview0.getStaticSizes();
		auto subview1_sizes = op_subview1.getStaticSizes();
		auto src0_memref_shape = op_subview0.getSourceType().getShape();
		auto src1_memref_shape = op_subview1.getSourceType().getShape();

		size_t ndim = subview0_sizes.size();
		// args correctness check 
		bool ok = true;
		ok = ok && (subview1_sizes.size() == ndim);
		ok = ok && (src0_memref_shape.size() == ndim);
		ok = ok && (src0_memref_shape.size() == ndim);
		for(int j=0; j<ndim; j++)
		{
			auto size_j_0 = op_subview0.getStaticSize(j);
			auto size_j_1 = op_subview1.getStaticSize(j);
			ok = ok && (size_j_0 == size_j_1);
			ok = ok && (size_j_0 == 1 || j == ndim-1);
		}
		if(!ok)
			assert(0 && "subview dimension do not match error");
		
		// data type string 
		auto data_type = op_subview0.getSourceType().getElementType();
		if(data_type.isF32()) type_string = "float";
		else if(data_type.isInteger(32)) type_string = "int";
		else if(data_type.isInteger(64)) type_string = "long";
		else assert("unsupported vector data type" && 0);

		int vec_len = op_subview0.getStaticSize(ndim-1);
		if(vec_len == 1) type_string = type_string + "*";
		else if(vec_len == 2) type_string = type_string + "2*";
		else if(vec_len == 3) type_string = type_string + "3*";
		else if(vec_len == 4) type_string = type_string + "4*";
		else assert("unsupported vector length" && 0);
		
		std::string vcp_var0_string, vcp_var1_string;
		mlir::Value subview0_src = op_subview0.getViewSource();
		mlir::Value subview1_src = op_subview1.getViewSource();
		int var0_idx = search_var_idx(subview0_src, variables_dict);
		int var1_idx = search_var_idx(subview1_src, variables_dict);
		if(var0_idx == -1 || var1_idx == -1)
			assert(0 && "error, can't find vec copy source var\n");
		else
		{
			vcp_var0_string = variables_dict[var0_idx].second;
			vcp_var1_string = variables_dict[var1_idx].second;
		}
		
		// pointer index string
		std::string offset_string0 = "";
		std::string offset_string1 = "";
		auto subview0_offsets = op_subview0.getStaticOffsets();
		auto subview1_offsets = op_subview1.getStaticOffsets();

		ok = ok && (op_subview0.getStaticOffsets().size() == ndim);
		ok = ok && (op_subview1.getStaticOffsets().size() == ndim);
		if(!ok)
			assert(0 && "subview dimension do not match error");

		for(int i=0; i<ndim; i++)
		{
			std::string var_string0;
			std::string var_string1;

			if(op_subview0.isDynamicOffset(i))
			{
				auto tmp_value = op_subview0.getDynamicOffset(i);
				int tmp_idx = search_var_idx(tmp_value, variables_dict);
				if(tmp_idx == -1)
					printf("error!, can't find Dynamic offset var\n");
				else
					var_string0 = variables_dict[tmp_idx].second;
			}
			else
			{
				int static_offset_i = op_subview0.getStaticOffset(i);
				var_string0 = std::to_string(static_offset_i);
			}
			
			if(op_subview1.isDynamicOffset(i))
			{
				auto tmp_value = op_subview1.getDynamicOffset(i);
				int tmp_idx = search_var_idx(tmp_value, variables_dict);
				if(tmp_idx == -1)
					printf("error!, can't find Dynamic offset var\n");
				else
					var_string1 = variables_dict[tmp_idx].second;
			}
			else
			{
				int static_offset_i = op_subview1.getStaticOffset(i);
				var_string1 = std::to_string(static_offset_i);
			}

			for(int j=i+1; j<ndim; j++)
			{
				var_string0 = var_string0 + "*" + std::to_string(src0_memref_shape[j]);
				var_string1 = var_string1 + "*" + std::to_string(src1_memref_shape[j]);
			}

			offset_string0 = offset_string0 + " + " + var_string0;
			offset_string1 = offset_string1 + " + " + var_string1;
		}

		std::string string_0 = "*(" + type_string + ")" + "(" + vcp_var0_string + offset_string0 + ")";
		std::string string_1 = "*(" + type_string + ")" + "(" + vcp_var1_string + offset_string1 + ")";

		if(flip)
			result = string_0 + "=" + string_1 + ";";
		else
			result = string_1 + "=" + string_0 + ";";

		return result;
	}

	std::string Ele_Parser::parse_load_op(mlir::memref::LoadOp& load_op, std::vector<std::pair<mlir::Value, std::string>>& variables_dict)
	{
		std::string op_load_string;
		auto source_memref = load_op.getMemRef();
		int source_memref_idx = search_var_idx(source_memref, variables_dict);
		assert(source_memref_idx != -1);
		auto source_name = variables_dict[source_memref_idx].second;

		auto result_memref_type = source_memref.getType();
		std::string result_type = parse_variable_type(result_memref_type, false, false);

		std::string result_name;
		auto load_op_result = load_op.getResult();
		int result_idx = search_var_idx(load_op_result, variables_dict);
		if(result_idx == -1)
		{
			int var_idx = variables_dict.size();
			result_name = "var" + std::to_string(var_idx);
			variables_dict.push_back({load_op_result, result_name});
			op_load_string = result_type + result_name + " = ";
		}
		else
		{
			result_name = variables_dict[result_idx].second;
			op_load_string = result_name + " = ";
		}

		op_load_string = op_load_string + source_name;

		std::string indices_string = "[";
		auto inds = load_op.getIndices();
		auto memref_shape = load_op.getMemRefType().getShape();
		for(int i=0; i<inds.size(); i++)
		{
			if(i!=0)
				indices_string = indices_string + " + ";
			mlir::Value ind = inds[i];
			int ind_idx = search_var_idx(ind, variables_dict);
			assert(ind_idx != -1);
			auto ind_string = variables_dict[ind_idx].second;
			for(int j=i+1; j<memref_shape.size(); j++)
				ind_string = ind_string + "*" + std::to_string(memref_shape[j]);
			indices_string = indices_string + ind_string;
		}
		op_load_string = op_load_string + indices_string + "]" + ";";
		return op_load_string;
	}

	std::string Ele_Parser::parse_store_op(mlir::memref::StoreOp& store_op, std::vector<std::pair<mlir::Value, std::string>>& variables_dict)
	{
		std::string op_store_string;
		auto left_var = store_op.getMemRef();
		int left_var_idx = search_var_idx(left_var, variables_dict);

		assert(left_var_idx != -1);
		std::string left_var_name = variables_dict[left_var_idx].second;
		
		std::string indices_string = "[";
		auto inds = store_op.getIndices();
		auto memref_shape = store_op.getMemRefType().getShape();
		for(int i=0; i<inds.size(); i++)
		{
			if(i!=0)
				indices_string = indices_string + " + ";
			mlir::Value ind = inds[i];
			int ind_idx = search_var_idx(ind, variables_dict);
			assert(ind_idx != -1);
			auto ind_string = variables_dict[ind_idx].second;
			for(int j=i+1; j<memref_shape.size(); j++)
				ind_string = ind_string + "*" + std::to_string(memref_shape[j]);
			indices_string = indices_string + ind_string;
		}
		indices_string = indices_string + "]";

		op_store_string = left_var_name + indices_string + " = ";

		auto right_var = store_op.getValue();
		int right_var_idx = search_var_idx(right_var, variables_dict);
		assert(right_var_idx != -1);
		std::string right_var_name = variables_dict[right_var_idx].second;
		
		op_store_string = op_store_string + right_var_name + ";";
		return op_store_string;
	}

}
