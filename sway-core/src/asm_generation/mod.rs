use std::{collections::HashMap, fmt};

use crate::type_engine::resolve_type;
use crate::{
    asm_generation::expression::convert_abi_fn_to_asm,
    asm_lang::{
        allocated_ops::{AllocatedOp, AllocatedRegister},
        virtual_register::*,
        Label, Op, OrganizationalOp, RealizedOp, VirtualImmediate12, VirtualImmediate24, VirtualOp,
    },
    error::*,
    parse_tree::Literal,
    semantic_analysis::{
        Namespace, TypedAstNode, TypedAstNodeContent, TypedDeclaration, TypedFunctionDeclaration,
        TypedParseTree,
    },
    types::ResolvedType,
    BuildConfig, Ident, TypeInfo,
};
use either::Either;

pub(crate) mod checks;
pub(crate) mod compiler_constants;
mod declaration;
mod expression;
mod finalized_asm;
mod register_sequencer;
mod while_loop;

pub(crate) use declaration::*;
pub(crate) use expression::*;
pub use finalized_asm::FinalizedAsm;
pub(crate) use register_sequencer::*;

use while_loop::convert_while_loop_to_asm;

// Initially, the bytecode will have a lot of individual registers being used. Each register will
// have a new unique identifier. For example, two separate invocations of `+` will result in 4
// registers being used for arguments and 2 for outputs.
//
// After that, the level 0 bytecode will go through a process where register use is minified,
// producing level 1 bytecode. This process is as such:
//
// 1. Detect the last time a register is read. After that, it can be reused and recycled to fit the
//    needs of the next "level 0 bytecode" register
//
// 2. Detect needless assignments and movements, and substitute registers in.
//    i.e.
//    a = b
//    c = a
//
//    would become
//    c = b
//
//
// After the level 1 bytecode is produced, level 2 bytecode is created by limiting the maximum
// number of registers and inserting bytecode to read from/write to memory where needed. Ideally,
// the algorithm for determining which registers will be written off to memory is based on how
// frequently that register is accessed in a particular section of code. Using this strategy, we
// hope to minimize memory writing.
//
// For each line, the number of times a virtual register is accessed between then and the end of the
// program is its register precedence. A virtual register's precedence is 0 if it is currently in
// "memory", and the above described number if it is not. This prevents over-prioritization of
// registers that have already been written off to memory.
//
/// The [HllAsmSet] contains either a contract ABI and corresponding ASM, a script's main
/// function's ASM, or a predicate's main function's ASM. ASM is never generated for libraries,
/// as that happens when the library itself is imported.
pub enum HllAsmSet<'sc> {
    ContractAbi {
        data_section: DataSection<'sc>,
        program_section: AbstractInstructionSet<'sc>,
    },
    ScriptMain {
        data_section: DataSection<'sc>,
        program_section: AbstractInstructionSet<'sc>,
    },
    PredicateMain {
        data_section: DataSection<'sc>,
        program_section: AbstractInstructionSet<'sc>,
    },
    // Libraries do not generate any asm.
    Library,
}

/// An [AbstractInstructionSet] is a set of instructions that use entirely virtual registers
/// and excessive moves, with the intention of later optimizing it.
#[derive(Clone)]
pub struct AbstractInstructionSet<'sc> {
    ops: Vec<Op<'sc>>,
}

/// "Realized" here refers to labels -- there are no more organizational
/// ops or labels. In this struct, they are all "realized" to offsets.
pub struct RealizedAbstractInstructionSet<'sc> {
    ops: Vec<RealizedOp<'sc>>,
}

impl<'sc> RealizedAbstractInstructionSet<'sc> {
    fn allocate_registers(self) -> InstructionSet<'sc> {
        // Eventually, we will use a cool graph-coloring algorithm.
        // For now, just keep a pool of registers and return
        // registers when they are not read anymore

        // construct a mapping from every op to the registers it uses
        let op_register_mapping = self
            .ops
            .into_iter()
            .map(|op| {
                (
                    op.clone(),
                    op.opcode.registers().into_iter().cloned().collect(),
                )
            })
            .collect::<Vec<_>>();

        // get registers from the pool.
        let mut pool = RegisterPool::init();
        let mut buf = vec![];
        for (ix, (op, _)) in op_register_mapping.iter().enumerate() {
            buf.push(AllocatedOp {
                opcode: op
                    .opcode
                    .allocate_registers(&mut pool, &op_register_mapping, ix),
                comment: op.comment.clone(),
                owning_span: op.owning_span.clone(),
            })
        }
        InstructionSet { ops: buf }
    }
}

/// An [InstructionSet] is produced by allocating registers on an [AbstractInstructionSet].
#[derive(Clone)]
pub struct InstructionSet<'sc> {
    ops: Vec<AllocatedOp<'sc>>,
}

type Data<'sc> = Literal<'sc>;
impl<'sc> AbstractInstructionSet<'sc> {
    /// Removes any jumps that jump to the subsequent line
    fn remove_sequential_jumps(&self) -> AbstractInstructionSet<'sc> {
        let mut buf = vec![];
        for i in 0..self.ops.len() - 1 {
            if let Op {
                opcode: Either::Right(OrganizationalOp::Jump(ref label)),
                ..
            } = self.ops[i]
            {
                if let Op {
                    opcode: Either::Right(OrganizationalOp::Label(ref label2)),
                    ..
                } = self.ops[i + 1]
                {
                    if label == label2 {
                        // this is a jump to the next line
                        // omit these by doing nothing
                        continue;
                    }
                }
            }
            buf.push(self.ops[i].clone());
        }
        // the last item cannot sequentially jump by definition so we add it in here
        if let Some(x) = self.ops.last() {
            buf.push(x.clone())
        };

        // scan through the jumps and remove any labels that are unused
        // this could of course be N instead of 2N if i did this in the above for loop.
        // However, the sweep for unused labels is inevitable regardless of the above phase
        // so might as well do it here.
        let mut buf2 = vec![];
        for op in &buf {
            match op.opcode {
                Either::Right(OrganizationalOp::Label(ref label)) => {
                    if label_is_used(&buf, label) {
                        buf2.push(op.clone());
                    }
                }
                _ => buf2.push(op.clone()),
            }
        }

        AbstractInstructionSet { ops: buf2 }
    }

    /// Runs two passes -- one to get the instruction offsets of the labels
    /// and one to replace the labels in the organizational ops
    fn realize_labels(
        self,
        data_section: &DataSection<'sc>,
    ) -> RealizedAbstractInstructionSet<'sc> {
        let mut label_namespace: HashMap<&Label, u64> = Default::default();
        let mut counter = 0;
        for op in &self.ops {
            match op.opcode {
                Either::Right(OrganizationalOp::Label(ref lab)) => {
                    label_namespace.insert(lab, counter);
                }
                // A special case for LWDataId which may be 1 or 2 ops, depending on the source size.
                Either::Left(VirtualOp::LWDataId(_, ref data_id)) => {
                    let type_of_data = data_section.type_of_data(data_id).expect(
                        "Internal miscalculation in data section -- data id did not match up to any actual data",
                    );
                    counter += if type_of_data.stack_size_of() > 1 {
                        2
                    } else {
                        1
                    };
                }
                // these ops will end up being exactly one op, so the counter goes up one
                Either::Right(OrganizationalOp::Jump(..))
                | Either::Right(OrganizationalOp::JumpIfNotEq(..))
                | Either::Left(_) => {
                    counter += 1;
                }
                Either::Right(OrganizationalOp::Comment) => (),
                Either::Right(OrganizationalOp::DataSectionOffsetPlaceholder) => {
                    // If the placeholder is 32 bits, this is 1. if 64, this should be 2. We use LW
                    // to load the data, which loads a whole word, so for now this is 2.
                    counter += 2
                }
            }
        }

        let mut realized_ops = vec![];
        for Op {
            opcode,
            owning_span,
            comment,
        } in self.ops.clone().into_iter()
        {
            match opcode {
                Either::Left(op) => realized_ops.push(RealizedOp {
                    opcode: op,
                    owning_span,
                    comment,
                }),
                Either::Right(org_op) => match org_op {
                    OrganizationalOp::Jump(ref lab) => {
                        let offset = label_namespace.get(lab).unwrap();
                        let imm = VirtualImmediate24::new_unchecked(
                            *offset,
                            "Programs with more than 2^24 labels are unsupported right now",
                        );
                        realized_ops.push(RealizedOp {
                            opcode: VirtualOp::JI(imm),
                            owning_span,
                            comment,
                        });
                    }
                    OrganizationalOp::JumpIfNotEq(r1, r2, ref lab) => {
                        let offset = label_namespace.get(lab).unwrap();
                        let imm = VirtualImmediate12::new_unchecked(
                            *offset,
                            "Programs with more than 2^12 labels are unsupported right now",
                        );
                        realized_ops.push(RealizedOp {
                            opcode: VirtualOp::JNEI(r1, r2, imm),
                            owning_span,
                            comment,
                        });
                    }
                    OrganizationalOp::DataSectionOffsetPlaceholder => {
                        realized_ops.push(RealizedOp {
                            opcode: VirtualOp::DataSectionOffsetPlaceholder,
                            owning_span: None,
                            comment: String::new(),
                        });
                    }
                    OrganizationalOp::Comment => continue,
                    OrganizationalOp::Label(..) => continue,
                },
            };
        }
        RealizedAbstractInstructionSet { ops: realized_ops }
    }
}

#[derive(Debug)]
struct RegisterAllocationStatus {
    reg: AllocatedRegister,
    in_use: Option<VirtualRegister>,
}
#[derive(Debug)]
pub(crate) struct RegisterPool {
    registers: Vec<RegisterAllocationStatus>,
}

impl RegisterPool {
    fn init() -> Self {
        let register_pool: Vec<RegisterAllocationStatus> = (0
            // - 1 because we reserve the final register for the data_section begin
            ..compiler_constants::NUM_ALLOCATABLE_REGISTERS)
            .map(|x| RegisterAllocationStatus {
                reg: AllocatedRegister::Allocated(x),
                in_use: None,
            })
            .collect();
        Self {
            registers: register_pool,
        }
    }

    /// Checks if any currently used registers are no longer in use, updates the pool,
    /// and grabs an available register.
    pub(crate) fn get_register(
        &mut self,
        virtual_register: &VirtualRegister,
        op_register_mapping: &[(RealizedOp, std::collections::HashSet<VirtualRegister>)],
    ) -> Option<AllocatedRegister> {
        // check if this register has already been allocated for
        if let a @ Some(_) = self.registers.iter().find_map(
            |RegisterAllocationStatus { reg, in_use }| match in_use {
                Some(x) if x == virtual_register => Some(reg),
                _ => None,
            },
        ) {
            return a.cloned();
        }

        // scan to see if any of the old ones are no longer in use
        for RegisterAllocationStatus { in_use, .. } in
            self.registers.iter_mut().filter(|r| r.in_use.is_some())
        {
            if virtual_register_is_never_accessed_again(
                in_use.as_ref().unwrap(),
                op_register_mapping,
            ) {
                *in_use = None;
            }
        }
        // find the next unused register, return it, assign it
        let next_available = self
            .registers
            .iter_mut()
            .find(|RegisterAllocationStatus { in_use, .. }| in_use.is_none());

        match next_available {
            Some(RegisterAllocationStatus { in_use, reg }) => {
                *in_use = Some(virtual_register.clone());
                Some(reg.clone())
            }
            None => None,
        }
    }
}

fn virtual_register_is_never_accessed_again(
    reg: &VirtualRegister,
    ops: &[(RealizedOp, std::collections::HashSet<VirtualRegister>)],
) -> bool {
    !ops.iter().any(|(_, regs)| regs.contains(reg))
}

/// helper function to check if a label is used in a given buffer of ops
fn label_is_used<'sc>(buf: &[Op<'sc>], label: &Label) -> bool {
    buf.iter().any(|Op { ref opcode, .. }| match opcode {
        Either::Right(OrganizationalOp::Jump(ref l)) if label == l => true,
        Either::Right(OrganizationalOp::JumpIfNotEq(_reg0, _reg1, ref l)) if label == l => true,
        _ => false,
    })
}

#[derive(Default, Clone, Debug)]
pub struct DataSection<'sc> {
    /// the data to be put in the data section of the asm
    pub value_pairs: Vec<Data<'sc>>,
}

impl<'sc> DataSection<'sc> {
    /// Given a [DataId], calculate the offset _from the beginning of the data section_ to the data
    /// in bytes.
    pub(crate) fn offset_to_id(&self, id: &DataId) -> usize {
        self.value_pairs
            .iter()
            .take(id.0 as usize)
            .map(|x| x.to_bytes().len())
            .sum()
    }

    pub(crate) fn serialize_to_bytes(&self) -> Vec<u8> {
        // not the exact right capacity but serves as a lower bound
        let mut buf = Vec::with_capacity(self.value_pairs.len());
        for val in &self.value_pairs {
            buf.append(&mut val.to_bytes().to_vec());
        }
        buf
    }

    /// Calculates the return type of the data held at a specific [DataId].
    pub(crate) fn type_of_data(&self, id: &DataId) -> Option<ResolvedType<'sc>> {
        self.value_pairs.get(id.0 as usize).map(|x| x.as_type())
    }

    /// When generating code, sometimes a hard-coded data pointer is needed to reference
    /// static values that have a length longer than one word.
    /// This method appends pointers to the end of the data section (thus, not altering the data
    /// offsets of previous data).
    /// `pointer_value` is in _bytes_ and refers to the offset from instruction start to the data
    /// in question.
    pub(crate) fn append_pointer(&mut self, pointer_value: u64) -> DataId {
        let pointer_as_data = Literal::new_pointer_literal(pointer_value);
        self.insert_data_value(&pointer_as_data)
    }

    /// Given any data in the form of a [Literal] (using this type mainly because it includes type
    /// information and debug spans), insert it into the data section and return its offset as a
    /// [DataId].
    pub(crate) fn insert_data_value(&mut self, data: &Literal<'sc>) -> DataId {
        // if there is an identical data value, use the same id
        match self.value_pairs.iter().position(|x| x == data) {
            Some(num) => DataId(num as u32),
            None => {
                self.value_pairs.push(data.clone());
                // the index of the data section where the value is stored
                DataId((self.value_pairs.len() - 1) as u32)
            }
        }
    }
}

impl fmt::Display for DataSection<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut data_buf = String::new();
        for (ix, data) in self.value_pairs.iter().enumerate() {
            let data_val = match data {
                Literal::U8(num) => format!(".u8 {:#04x}", num),
                Literal::U16(num) => format!(".u16 {:#04x}", num),
                Literal::U32(num) => format!(".u32 {:#04x}", num),
                Literal::U64(num) => format!(".u64 {:#04x}", num),
                Literal::Boolean(b) => format!(".bool {}", if *b { "0x01" } else { "0x00" }),
                Literal::String(st) => format!(".str \"{}\"", st),
                Literal::Byte(b) => format!(".byte {:#08b}", b),
                Literal::B256(b) => format!(
                    ".b256 0x{}",
                    b.iter()
                        .map(|x| format!("{:02x}", x))
                        .collect::<Vec<_>>()
                        .join("")
                ),
            };
            let data_label = DataId(ix as u32);
            data_buf.push_str(&format!("{} {}\n", data_label, data_val));
        }

        write!(f, ".data:\n{}", data_buf)
    }
}

impl fmt::Display for HllAsmSet<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HllAsmSet::ScriptMain {
                data_section,
                program_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            HllAsmSet::PredicateMain {
                data_section,
                program_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            HllAsmSet::ContractAbi {
                data_section,
                program_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            // Libraries do not directly generate any asm.
            HllAsmSet::Library => write!(f, ""),
        }
    }
}

impl fmt::Display for JumpOptimizedAsmSet<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JumpOptimizedAsmSet::ScriptMain {
                data_section,
                program_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            JumpOptimizedAsmSet::PredicateMain {
                data_section,
                program_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            JumpOptimizedAsmSet::ContractAbi {
                data_section,
                program_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            // Libraries do not directly generate any asm.
            JumpOptimizedAsmSet::Library => write!(f, ""),
        }
    }
}

impl<'sc> fmt::Display for RegisterAllocatedAsmSet<'sc> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegisterAllocatedAsmSet::ScriptMain {
                program_section,
                data_section,
            } => {
                write!(f, "{}\n{}", program_section, data_section)
            }
            RegisterAllocatedAsmSet::PredicateMain {
                program_section,
                data_section,
            } => {
                write!(f, "{}\n{}", program_section, data_section)
            }
            RegisterAllocatedAsmSet::ContractAbi {
                program_section,
                data_section,
            } => {
                write!(f, "{}\n{}", program_section, data_section)
            }
            // Libraries do not directly generate any asm.
            RegisterAllocatedAsmSet::Library => write!(f, ""),
        }
    }
}

impl<'sc> fmt::Display for FinalizedAsm<'sc> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FinalizedAsm::ScriptMain {
                program_section,
                data_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            FinalizedAsm::PredicateMain {
                program_section,
                data_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            FinalizedAsm::ContractAbi {
                program_section,
                data_section,
            } => write!(f, "{}\n{}", program_section, data_section),
            // Libraries do not directly generate any asm.
            FinalizedAsm::Library => write!(f, ""),
        }
    }
}

impl fmt::Display for AbstractInstructionSet<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            ".program:\n{}",
            self.ops
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

impl<'sc> fmt::Display for InstructionSet<'sc> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            ".program:\n{}",
            self.ops
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

#[derive(Default, Clone, Debug)]
pub(crate) struct AsmNamespace<'sc> {
    data_section: DataSection<'sc>,
    variables: HashMap<Ident<'sc>, VirtualRegister>,
}

/// An address which refers to a value in the data section of the asm.
#[derive(Clone, Debug)]
pub(crate) struct DataId(pub(crate) u32);

impl fmt::Display for DataId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "data_{}", self.0)
    }
}

impl<'sc> AsmNamespace<'sc> {
    pub(crate) fn insert_variable(
        &mut self,
        var_name: Ident<'sc>,
        register_location: VirtualRegister,
    ) {
        self.variables.insert(var_name, register_location);
    }
    pub(crate) fn insert_data_value(&mut self, data: &Data<'sc>) -> DataId {
        self.data_section.insert_data_value(data)
    }
    /// Finds the register which contains variable `var_name`
    /// The `get` is unwrapped, because invalid variable expressions are
    /// checked for in the type checking stage.
    pub(crate) fn look_up_variable(
        &self,
        var_name: &Ident<'sc>,
    ) -> CompileResult<'sc, &VirtualRegister> {
        match self.variables.get(var_name) {
            Some(o) => ok(o, vec![], vec![]),
            None => err(
                vec![],
                vec![CompileError::Internal(
                    "Unknown variable in assembly generation. This should have been an error \
                     during type checking.",
                    var_name.span.clone(),
                )],
            ),
        }
    }
}

pub(crate) fn compile_ast_to_asm<'sc>(
    ast: TypedParseTree<'sc>,
    build_config: &BuildConfig,
) -> CompileResult<'sc, FinalizedAsm<'sc>> {
    let mut register_sequencer = RegisterSequencer::new();
    let mut warnings = vec![];
    let mut errors = vec![];
    let (asm, _asm_namespace) = match ast {
        TypedParseTree::Script {
            main_function,
            namespace: ast_namespace,
            declarations,
            ..
        } => {
            let mut namespace: AsmNamespace = Default::default();
            let mut asm_buf = build_preamble(&mut register_sequencer).to_vec();
            check!(
                add_all_constant_decls(
                    &mut namespace,
                    &mut register_sequencer,
                    &mut asm_buf,
                    &declarations,
                    &ast_namespace,
                ),
                return err(warnings, errors),
                warnings,
                errors
            );
            // start generating from the main function
            let return_register = register_sequencer.next();
            let mut body = check!(
                convert_code_block_to_asm(
                    &main_function.body,
                    &mut namespace,
                    &mut register_sequencer,
                    // TODO validate that this isn't just implicit returns?
                    Some(&return_register),
                ),
                vec![],
                warnings,
                errors
            );
            asm_buf.append(&mut body);
            asm_buf.append(&mut check!(
                ret_or_retd_value(
                    &main_function,
                    return_register,
                    &mut register_sequencer,
                    &mut namespace
                ),
                return err(warnings, errors),
                warnings,
                errors
            ));

            (
                HllAsmSet::ScriptMain {
                    program_section: AbstractInstructionSet { ops: asm_buf },
                    data_section: namespace.data_section.clone(),
                },
                namespace,
            )
        }
        TypedParseTree::Predicate {
            main_function,
            namespace: ast_namespace,
            declarations,
            ..
        } => {
            let mut namespace: AsmNamespace = Default::default();
            let mut asm_buf = build_preamble(&mut register_sequencer).to_vec();
            check!(
                add_all_constant_decls(
                    &mut namespace,
                    &mut register_sequencer,
                    &mut asm_buf,
                    &declarations,
                    &ast_namespace,
                ),
                return err(warnings, errors),
                warnings,
                errors
            );
            // start generating from the main function
            let mut body = check!(
                convert_code_block_to_asm(
                    &main_function.body,
                    &mut namespace,
                    &mut register_sequencer,
                    None,
                ),
                vec![],
                warnings,
                errors
            );
            asm_buf.append(&mut body);

            (
                HllAsmSet::PredicateMain {
                    program_section: AbstractInstructionSet { ops: asm_buf },
                    data_section: namespace.data_section.clone(),
                },
                namespace,
            )
        }
        TypedParseTree::Contract {
            abi_entries,
            namespace: ast_namespace,
            declarations,
            ..
        } => {
            let mut namespace: AsmNamespace = Default::default();
            let mut asm_buf = build_preamble(&mut register_sequencer).to_vec();
            check!(
                add_all_constant_decls(
                    &mut namespace,
                    &mut register_sequencer,
                    &mut asm_buf,
                    &declarations,
                    &ast_namespace,
                ),
                return err(warnings, errors),
                warnings,
                errors
            );
            let (selectors_and_labels, mut contract_asm) = check!(
                compile_contract_to_selectors(abi_entries, &mut namespace, &mut register_sequencer),
                return err(warnings, errors),
                warnings,
                errors
            );
            asm_buf.append(&mut build_contract_abi_switch(
                &mut register_sequencer,
                &mut namespace.data_section,
                selectors_and_labels,
            ));
            asm_buf.append(&mut contract_asm);

            (
                HllAsmSet::ContractAbi {
                    program_section: AbstractInstructionSet { ops: asm_buf },
                    data_section: namespace.data_section.clone(),
                },
                namespace,
            )
        }
        TypedParseTree::Library { .. } => (HllAsmSet::Library, Default::default()),
    };

    if build_config.print_intermediate_asm {
        println!("{}", asm);
    }

    let finalized_asm = asm
        .remove_unnecessary_jumps()
        .allocate_registers()
        .optimize();

    if build_config.print_finalized_asm {
        println!("{}", finalized_asm);
    }

    check!(
        crate::checks::check_invalid_opcodes(&finalized_asm),
        return err(warnings, errors),
        warnings,
        errors
    );

    ok(finalized_asm, warnings, errors)
}

impl<'sc> HllAsmSet<'sc> {
    pub(crate) fn remove_unnecessary_jumps(self) -> JumpOptimizedAsmSet<'sc> {
        match self {
            HllAsmSet::ScriptMain {
                data_section,
                program_section,
            } => JumpOptimizedAsmSet::ScriptMain {
                data_section,
                program_section: program_section.remove_sequential_jumps(),
            },
            HllAsmSet::PredicateMain {
                data_section,
                program_section,
            } => JumpOptimizedAsmSet::PredicateMain {
                data_section,
                program_section: program_section.remove_sequential_jumps(),
            },
            HllAsmSet::Library {} => JumpOptimizedAsmSet::Library,
            HllAsmSet::ContractAbi {
                data_section,
                program_section,
            } => JumpOptimizedAsmSet::ContractAbi {
                data_section,
                program_section: program_section.remove_sequential_jumps(),
            },
        }
    }
}

impl<'sc> JumpOptimizedAsmSet<'sc> {
    fn allocate_registers(self) -> RegisterAllocatedAsmSet<'sc> {
        match self {
            JumpOptimizedAsmSet::Library => RegisterAllocatedAsmSet::Library,
            JumpOptimizedAsmSet::ScriptMain {
                data_section,
                program_section,
            } => {
                let program_section = program_section
                    .realize_labels(&data_section)
                    .allocate_registers();
                RegisterAllocatedAsmSet::ScriptMain {
                    data_section,
                    program_section,
                }
            }
            JumpOptimizedAsmSet::PredicateMain {
                data_section,
                program_section,
            } => {
                let program_section = program_section
                    .realize_labels(&data_section)
                    .allocate_registers();
                RegisterAllocatedAsmSet::PredicateMain {
                    data_section,
                    program_section,
                }
            }
            JumpOptimizedAsmSet::ContractAbi {
                program_section,
                data_section,
            } => RegisterAllocatedAsmSet::ContractAbi {
                program_section: program_section
                    .realize_labels(&data_section)
                    .allocate_registers(),
                data_section,
            },
        }
    }
}

/// Represents an ASM set which has had jump labels and jumps optimized
pub enum JumpOptimizedAsmSet<'sc> {
    ContractAbi {
        data_section: DataSection<'sc>,
        program_section: AbstractInstructionSet<'sc>,
    },
    ScriptMain {
        data_section: DataSection<'sc>,
        program_section: AbstractInstructionSet<'sc>,
    },
    PredicateMain {
        data_section: DataSection<'sc>,
        program_section: AbstractInstructionSet<'sc>,
    },
    // Libraries do not generate any asm.
    Library,
}
/// Represents an ASM set which has had registers allocated
pub enum RegisterAllocatedAsmSet<'sc> {
    ContractAbi {
        data_section: DataSection<'sc>,
        program_section: InstructionSet<'sc>,
    },
    ScriptMain {
        data_section: DataSection<'sc>,
        program_section: InstructionSet<'sc>,
    },
    PredicateMain {
        data_section: DataSection<'sc>,
        program_section: InstructionSet<'sc>,
    },
    // Libraries do not generate any asm.
    Library,
}

impl<'sc> RegisterAllocatedAsmSet<'sc> {
    fn optimize(self) -> FinalizedAsm<'sc> {
        // TODO implement this -- noop for now
        match self {
            RegisterAllocatedAsmSet::Library => FinalizedAsm::Library,
            RegisterAllocatedAsmSet::ScriptMain {
                mut program_section,
                data_section,
            } => {
                // ensure there's an even number of ops so the
                // data section offset is valid
                if program_section.ops.len() & 1 != 0 {
                    program_section.ops.push(AllocatedOp {
                        opcode: crate::asm_lang::allocated_ops::AllocatedOpcode::NOOP,
                        comment: "word-alignment of data section".into(),
                        owning_span: None,
                    });
                }
                FinalizedAsm::ScriptMain {
                    program_section,
                    data_section,
                }
            }
            RegisterAllocatedAsmSet::PredicateMain {
                mut program_section,
                data_section,
            } => {
                // ensure there's an even number of ops so the
                // data section offset is valid
                if program_section.ops.len() & 1 != 0 {
                    program_section.ops.push(AllocatedOp {
                        opcode: crate::asm_lang::allocated_ops::AllocatedOpcode::NOOP,
                        comment: "word-alignment of data section".into(),
                        owning_span: None,
                    });
                }
                FinalizedAsm::PredicateMain {
                    program_section,
                    data_section,
                }
            }
            RegisterAllocatedAsmSet::ContractAbi {
                mut program_section,
                data_section,
            } => {
                // ensure there's an even number of ops so the
                // data section offset is valid
                if program_section.ops.len() & 1 != 0 {
                    program_section.ops.push(AllocatedOp {
                        opcode: crate::asm_lang::allocated_ops::AllocatedOpcode::NOOP,
                        comment: "word-alignment of data section".into(),
                        owning_span: None,
                    });
                }
                FinalizedAsm::ContractAbi {
                    program_section,
                    data_section,
                }
            }
        }
    }
}

pub(crate) enum NodeAsmResult<'sc> {
    JustAsm(Vec<Op<'sc>>),
    ReturnStatement { asm: Vec<Op<'sc>> },
}

/// The tuple being returned here contains the opcodes of the code block and,
/// optionally, a return register in case this node was a return statement
fn convert_node_to_asm<'sc>(
    node: &TypedAstNode<'sc>,
    namespace: &mut AsmNamespace<'sc>,
    register_sequencer: &mut RegisterSequencer,
    // Where to put the return value of this node, if it is needed.
    return_register: Option<&VirtualRegister>,
) -> CompileResult<'sc, NodeAsmResult<'sc>> {
    let mut warnings = vec![];
    let mut errors = vec![];
    match &node.content {
        TypedAstNodeContent::WhileLoop(r#loop) => {
            let res = check!(
                convert_while_loop_to_asm(r#loop, namespace, register_sequencer),
                return err(warnings, errors),
                warnings,
                errors
            );
            ok(NodeAsmResult::JustAsm(res), warnings, errors)
        }
        TypedAstNodeContent::Declaration(typed_decl) => {
            let res = check!(
                convert_decl_to_asm(typed_decl, namespace, register_sequencer),
                return err(warnings, errors),
                warnings,
                errors
            );
            ok(NodeAsmResult::JustAsm(res), warnings, errors)
        }
        TypedAstNodeContent::ImplicitReturnExpression(exp) => {
            // if a return register was specified, we use it. If not, we generate a register but
            // it is going to get thrown away later (in coalescing) as it is never read
            let return_register = if let Some(return_register) = return_register {
                return_register.clone()
            } else {
                register_sequencer.next()
            };
            let ops = check!(
                convert_expression_to_asm(exp, namespace, &return_register, register_sequencer),
                return err(warnings, errors),
                warnings,
                errors
            );
            ok(
                NodeAsmResult::ReturnStatement { asm: ops },
                warnings,
                errors,
            )
        }
        TypedAstNodeContent::ReturnStatement(exp) => {
            // if a return register was specified, we use it. If not, we generate a register but
            // it is going to get thrown away later (in coalescing) as it is never read
            let return_register = if let Some(return_register) = return_register {
                return_register.clone()
            } else {
                register_sequencer.next()
            };
            let ops = check!(
                convert_expression_to_asm(
                    &exp.expr,
                    namespace,
                    &return_register,
                    register_sequencer
                ),
                return err(warnings, errors),
                warnings,
                errors
            );
            ok(
                NodeAsmResult::ReturnStatement { asm: ops },
                warnings,
                errors,
            )
        }
        TypedAstNodeContent::Expression(ref typed_expr) => {
            let return_register = if let Some(return_register) = return_register {
                return_register.clone()
            } else {
                register_sequencer.next()
            };
            let asm = check!(
                convert_expression_to_asm(
                    typed_expr,
                    namespace,
                    &return_register,
                    register_sequencer
                ),
                return err(warnings, errors),
                warnings,
                errors
            );
            ok(NodeAsmResult::JustAsm(asm), warnings, errors)
        }
        a => {
            println!("Unimplemented: {:?}", a);
            errors.push(CompileError::Unimplemented(
                "The ASM for this construct has not been written yet.",
                node.clone().span,
            ));
            err(warnings, errors)
        }
    }
}

/// Builds the asm preamble, which includes metadata and a jump past the metadata.
/// Right now, it looks like this:
///
/// WORD OP
/// 1    JI program_start
/// -    NOOP
/// 2    DATA_START (0-32) (in bytes, offset from $is)
/// -    DATA_START (32-64)
/// 3    LW $ds $is               1 (where 1 is in words and $is is a byte address to base off of)
/// -    ADD $ds $ds $is
/// 4    .program_start:
fn build_preamble(register_sequencer: &mut RegisterSequencer) -> [Op<'static>; 6] {
    let label = register_sequencer.get_label();
    [
        // word 1
        Op::jump_to_label(label.clone()),
        // word 1.5
        Op {
            opcode: Either::Left(VirtualOp::NOOP),
            comment: "".into(),
            owning_span: None,
        },
        // word 2 -- full word u64 placeholder
        Op {
            opcode: Either::Right(OrganizationalOp::DataSectionOffsetPlaceholder),
            comment: "data section offset".into(),
            owning_span: None,
        },
        Op::unowned_jump_label_comment(label, "end of metadata"),
        // word 3 -- load the data offset into $ds
        Op {
            opcode: Either::Left(VirtualOp::DataSectionRegisterLoadPlaceholder),
            comment: "".into(),
            owning_span: None,
        },
        // word 3.5 -- add $ds $ds $is
        Op {
            opcode: Either::Left(VirtualOp::ADD(
                VirtualRegister::Constant(ConstantRegister::DataSectionStart),
                VirtualRegister::Constant(ConstantRegister::DataSectionStart),
                VirtualRegister::Constant(ConstantRegister::InstructionStart),
            )),
            comment: "".into(),
            owning_span: None,
        },
    ]
}

/// Builds the contract switch statement, or function selector, which takes the selector
/// stored in the call frame (see https://github.com/FuelLabs/sway/issues/97#issuecomment-870150684
/// for an explanation of its location)
fn build_contract_abi_switch<'sc>(
    register_sequencer: &mut RegisterSequencer,
    data_section: &mut DataSection<'sc>,
    selectors_and_labels: Vec<([u8; 4], Label)>,
) -> Vec<Op<'sc>> {
    let input_selector_register = register_sequencer.next();
    let mut asm_buf = vec![Op {
        opcode: Either::Right(OrganizationalOp::Comment),
        comment: "Begin contract ABI selector switch".into(),
        owning_span: None,
    }];
    // load the selector from the call frame
    asm_buf.push(Op {
        opcode: Either::Left(VirtualOp::LW(
            input_selector_register.clone(),
            VirtualRegister::Constant(ConstantRegister::FramePointer),
            // see https://github.com/FuelLabs/fuel-specs/pull/193#issuecomment-876496372
            // We expect the last four bytes of this word to contain the selector, and the first
            // four bytes to all be 0.
            VirtualImmediate12::new_unchecked(73, "constant infallible value"),
        )),
        comment: "load input function selector".into(),
        owning_span: None,
    });

    for (selector, label) in selectors_and_labels {
        // put the selector in the data section
        let data_label =
            data_section.insert_data_value(&Literal::U32(u32::from_be_bytes(selector)));
        // load the data into a register for comparison
        let prog_selector_register = register_sequencer.next();
        asm_buf.push(Op {
            opcode: Either::Left(VirtualOp::LWDataId(
                prog_selector_register.clone(),
                data_label,
            )),
            comment: "load fn selector for comparison".into(),
            owning_span: None,
        });
        // compare with the input selector
        let comparison_result_register = register_sequencer.next();
        asm_buf.push(Op {
            opcode: Either::Left(VirtualOp::EQ(
                comparison_result_register.clone(),
                input_selector_register.clone(),
                prog_selector_register,
            )),
            comment: "function selector comparison".into(),
            owning_span: None,
        });

        // jump to the function label if the selector was equal
        asm_buf.push(Op {
            // if the comparison result is _not_ equal to 0, then it was indeed equal.
            opcode: Either::Right(OrganizationalOp::JumpIfNotEq(
                VirtualRegister::Constant(ConstantRegister::Zero),
                comparison_result_register,
                label,
            )),
            comment: "jump to selected function".into(),
            owning_span: None,
        });
    }

    // if none of the selectors matched, then ret
    asm_buf.push(Op {
        // see https://github.com/FuelLabs/sway/issues/97#issuecomment-875674105
        opcode: Either::Left(VirtualOp::RET(VirtualRegister::Constant(
            ConstantRegister::Zero,
        ))),
        comment: "return if no selectors matched".into(),
        owning_span: None,
    });

    asm_buf
}

fn add_all_constant_decls<'sc>(
    namespace: &mut AsmNamespace<'sc>,
    register_sequencer: &mut RegisterSequencer,
    asm_buf: &mut Vec<Op<'sc>>,
    declarations: &[TypedDeclaration<'sc>],
    ast_namespace: &Namespace<'sc>,
) -> CompileResult<'sc, ()> {
    let mut warnings = vec![];
    let mut errors = vec![];
    check!(
        add_global_constant_decls(namespace, register_sequencer, asm_buf, declarations),
        return err(warnings, errors),
        warnings,
        errors
    );
    check!(
        add_module_constant_decls(namespace, register_sequencer, asm_buf, ast_namespace),
        return err(warnings, errors),
        warnings,
        errors
    );
    ok((), warnings, errors)
}

fn add_global_constant_decls<'sc>(
    namespace: &mut AsmNamespace<'sc>,
    register_sequencer: &mut RegisterSequencer,
    asm_buf: &mut Vec<Op<'sc>>,
    declarations: &[TypedDeclaration<'sc>],
) -> CompileResult<'sc, ()> {
    let mut warnings = vec![];
    let mut errors = vec![];
    for declaration in declarations {
        if let TypedDeclaration::ConstantDeclaration(decl) = declaration {
            let mut ops = check!(
                convert_constant_decl_to_asm(decl, namespace, register_sequencer),
                return err(warnings, errors),
                warnings,
                errors
            );
            asm_buf.append(&mut ops);
        }
    }
    ok((), warnings, errors)
}

fn add_module_constant_decls<'sc>(
    namespace: &mut AsmNamespace<'sc>,
    register_sequencer: &mut RegisterSequencer,
    asm_buf: &mut Vec<Op<'sc>>,
    ast_namespace: &Namespace<'sc>,
) -> CompileResult<'sc, ()> {
    let mut warnings = vec![];
    let mut errors = vec![];

    // NOTE: this is currently flattening out the entire namespace, which is problematic.  To fix
    // it we need to support hierarchical names (or at least absolute normalised names) to
    // AsmNamespace.  This can be done in the new ASM generator which translates from IR, coming
    // soon.
    for ns in ast_namespace.get_all_imported_modules() {
        for decl in ns.get_all_declared_symbols() {
            if let TypedDeclaration::ConstantDeclaration(decl) = decl {
                let mut ops = check!(
                    convert_constant_decl_to_asm(decl, namespace, register_sequencer),
                    return err(warnings, errors),
                    warnings,
                    errors
                );
                asm_buf.append(&mut ops);
            }
        }
        check!(
            add_module_constant_decls(namespace, register_sequencer, asm_buf, ns),
            return err(warnings, errors),
            warnings,
            errors
        );
    }

    ok((), warnings, errors)
}

/// The function selector value and corresponding label.
type JumpDestination = Vec<([u8; 4], Label)>;
/// A vector of opcodes representing the body of a contract ABI function.
type AbiFunctionOpcodeBuffer<'sc> = Vec<Op<'sc>>;
/// The function selector information and compiled body of a contract ABI function.
type SerializedAbiFunction<'sc> = (JumpDestination, AbiFunctionOpcodeBuffer<'sc>);

/// Given a contract's abi entries, compile them to jump destinations and an opcode buffer.
fn compile_contract_to_selectors<'sc>(
    abi_entries: Vec<TypedFunctionDeclaration<'sc>>,
    namespace: &mut AsmNamespace<'sc>,
    register_sequencer: &mut RegisterSequencer,
) -> CompileResult<'sc, SerializedAbiFunction<'sc>> {
    let mut warnings = vec![];
    let mut errors = vec![];
    // for every ABI function, we need:
    // 0) a jump label
    // 1) loading the argument from the call frame into the register for the function
    // 2) the function's bytecode itself
    // 3) the function selector
    let mut selectors_labels_buf = vec![];
    let mut asm_buf = vec![];
    for decl in abi_entries {
        // TODO wrapping things in a struct should be doable by the compiler eventually,
        // allowing users to pass in any number of free-floating parameters (bound by immediate limits maybe).
        // https://github.com/FuelLabs/sway/pull/115#discussion_r666466414
        if decl.parameters.len() != 4 {
            errors.push(CompileError::InvalidNumberOfAbiParams {
                span: decl.parameters_span(),
            });
            continue;
        }
        // there are currently four parameters to every ABI function, and they are required to be
        // in this order
        let cgas_name = decl.parameters[0].name.clone();
        let bal_name = decl.parameters[1].name.clone();
        let coin_color_name = decl.parameters[2].name.clone();
        let user_argument_name = decl.parameters[3].name.clone();
        // the function selector is the first four bytes of the hashed declaration/params according
        // to https://github.com/FuelLabs/sway/issues/96
        let selector = check!(decl.to_fn_selector_value(), [0u8; 4], warnings, errors);
        let fn_label = register_sequencer.get_label();
        asm_buf.push(Op::jump_label(fn_label.clone(), decl.span.clone()));
        // load the call frame argument into the function argument register
        let user_argument_register = register_sequencer.next();
        let cgas_register = register_sequencer.next();
        let bal_register = register_sequencer.next();
        let coin_color_register = register_sequencer.next();
        asm_buf.push(load_user_argument(user_argument_register.clone()));
        asm_buf.push(load_cgas(cgas_register.clone()));
        asm_buf.push(load_bal(bal_register.clone()));
        asm_buf.push(load_coin_color(coin_color_register.clone()));

        asm_buf.append(&mut check!(
            convert_abi_fn_to_asm(
                &decl,
                (user_argument_name, user_argument_register),
                (cgas_name, cgas_register),
                (bal_name, bal_register),
                (coin_color_name, coin_color_register),
                namespace,
                register_sequencer
            ),
            vec![],
            warnings,
            errors
        ));
        selectors_labels_buf.push((selector, fn_label));
    }

    ok((selectors_labels_buf, asm_buf), warnings, errors)
}
/// Given a register, load the user-provided argument into it
fn load_user_argument<'sc>(return_register: VirtualRegister) -> Op<'sc> {
    Op {
        opcode: Either::Left(VirtualOp::LW(
            return_register,
            VirtualRegister::Constant(ConstantRegister::FramePointer),
            // see https://github.com/FuelLabs/fuel-specs/pull/193#issuecomment-876496372
            VirtualImmediate12::new_unchecked(74, "infallible constant 74"),
        )),
        comment: "loading argument into abi function".into(),
        owning_span: None,
    }
}
/// Given a register, load the current value of $cgas into it
fn load_cgas<'sc>(return_register: VirtualRegister) -> Op<'sc> {
    Op {
        opcode: Either::Left(VirtualOp::LW(
            return_register,
            VirtualRegister::Constant(ConstantRegister::ContextGas),
            VirtualImmediate12::new_unchecked(0, "infallible constant 0"),
        )),
        comment: "loading cgas into abi function".into(),
        owning_span: None,
    }
}
/// Given a register, load the current value of $bal into it
fn load_bal<'sc>(return_register: VirtualRegister) -> Op<'sc> {
    Op {
        opcode: Either::Left(VirtualOp::LW(
            return_register,
            VirtualRegister::Constant(ConstantRegister::Balance),
            VirtualImmediate12::new_unchecked(0, "infallible constant 0"),
        )),
        comment: "loading coin balance into abi function".into(),
        owning_span: None,
    }
}
/// Given a register, load a pointer to the current coin color into it
fn load_coin_color<'sc>(return_register: VirtualRegister) -> Op<'sc> {
    Op {
        opcode: Either::Left(VirtualOp::LW(
            return_register,
            VirtualRegister::Constant(ConstantRegister::FramePointer),
            VirtualImmediate12::new_unchecked(5, "infallible constant 5"),
        )),
        comment: "loading coin color into abi function".into(),
        owning_span: None,
    }
}

/// Given a [TypedFunctionDeclaration] and a `return_register`, return
/// the return value of the function using either a `RET` or a `RETD` opcode.
fn ret_or_retd_value<'sc>(
    func: &TypedFunctionDeclaration<'sc>,
    return_register: VirtualRegister,
    register_sequencer: &mut RegisterSequencer,
    namespace: &mut AsmNamespace<'sc>,
) -> CompileResult<'sc, Vec<Op<'sc>>> {
    let mut errors = vec![];
    let warnings = vec![];
    let mut asm_buf = vec![];
    let main_func_ret_ty: TypeInfo = match resolve_type(func.return_type, &func.return_type_span) {
        Ok(o) => o,
        Err(e) => {
            errors.push(e.into());
            return err(warnings, errors);
        }
    };

    if main_func_ret_ty == TypeInfo::Unit {
        // unit returns should always be zero, although because they can be
        // omitted from functions, the register is sometimes uninitialized.
        // Manually return zero in this case.
        return ok(
            vec![Op {
                opcode: Either::Left(VirtualOp::RET(VirtualRegister::Constant(
                    ConstantRegister::Zero,
                ))),
                owning_span: Some(func.return_type_span.clone()),
                comment: format!("fn {} returns unit", func.name.primary_name),
            }],
            warnings,
            errors,
        );
    }
    let span = crate::Span {
        span: pest::Span::new("TODO(static span)", 0, 0).unwrap(),
        path: None,
    };

    let size_of_main_func_return_bytes = main_func_ret_ty.size_in_words(&span).expect(
        "TODO(static span): Internal error: Static spans will allow for a proper error here.",
    ) * 8;
    if size_of_main_func_return_bytes <= 8 {
        asm_buf.push(Op {
            owning_span: None,
            opcode: Either::Left(VirtualOp::RET(return_register)),
            comment: format!("{} fn return value", func.name.primary_name),
        });
    } else {
        // if the type is larger than one word, then we use RETD to return data
        // RB is the size_in_bytes
        let rb_register = register_sequencer.next();
        let size_bytes = namespace.insert_data_value(&Literal::U64(size_of_main_func_return_bytes));
        // `return_register` is $rA
        asm_buf.push(Op {
            opcode: Either::Left(VirtualOp::LWDataId(rb_register.clone(), size_bytes)),
            owning_span: Some(func.return_type_span.clone()),
            comment: "loading rB for RETD".into(),
        });

        // now $rB has the size of the type in bytes
        asm_buf.push(Op {
            owning_span: None,
            opcode: Either::Left(VirtualOp::RETD(return_register, rb_register)),
            comment: format!("{} fn return value", func.name.primary_name),
        });
    }
    ok(asm_buf, warnings, errors)
}

// =================================================================================================
// Newer IR code gen.
//
// NOTE:  This is converting IR to Vec<Op> first, and then to finalized VM bytecode much like the
// original code above.  This is to keep things simple, and to reuse the current tools like
// DataSection.
//
// But this is not ideal and needs to be refactored:
// - AsmNamespace is tied to data structures from other stages like Ident and Literal.

use ir::*;

pub fn compile_ir_to_asm<'sc, 'ir>(
    ir: &'ir Context,
    build_config: &BuildConfig,
) -> CompileResult<'sc, FinalizedAsm<'sc>> {
    let mut warnings: Vec<CompileWarning<'sc>> = Vec::new();
    let mut errors: Vec<CompileError<'sc>> = Vec::new();

    let mut reg_seqr = RegisterSequencer::new();
    let mut bytecode: Vec<Op<'sc>> = build_preamble(&mut reg_seqr).to_vec();

    // Eventually when we get this 'correct' with no hacks we'll want to compile all the modules
    // separately and then use a linker to connect them.  This way we could also keep binary caches
    // of libraries and link against them, rather than recompile everything each time.
    assert!(ir.module_iter().count() == 1);
    let module = ir.module_iter().next().unwrap();
    let (data_section, mut ops, _) = check!(
        compile_module_to_asm(reg_seqr, ir, module),
        return err(warnings, errors),
        warnings,
        errors
    );
    bytecode.append(&mut ops);

    let asm = match module.get_kind(ir) {
        Kind::Script => HllAsmSet::ScriptMain {
            program_section: AbstractInstructionSet { ops: bytecode },
            data_section,
        },
        Kind::Contract => HllAsmSet::ContractAbi {
            program_section: AbstractInstructionSet { ops: bytecode },
            data_section,
        },
        Kind::Library | Kind::Predicate => todo!("libraries and predicates coming soon!"),
    };

    if build_config.print_intermediate_asm {
        println!("{}", asm);
    }

    let finalized_asm = asm
        .remove_unnecessary_jumps()
        .allocate_registers()
        .optimize();

    if build_config.print_finalized_asm {
        println!("{}", finalized_asm);
    }

    check!(
        crate::checks::check_invalid_opcodes(&finalized_asm),
        return err(warnings, errors),
        warnings,
        errors
    );

    ok(finalized_asm, warnings, errors)
}

fn compile_module_to_asm<'sc, 'ir>(
    reg_seqr: RegisterSequencer,
    context: &'ir Context,
    module: Module,
) -> CompileResult<'sc, (DataSection<'sc>, Vec<Op<'sc>>, RegisterSequencer)> {
    let mut builder = AsmBuilder::new(DataSection::default(), reg_seqr, context);
    match module.get_kind(context) {
        Kind::Script => {
            // We can't do function calls yet, so we expect everything to be inlined into `main`.
            let function = module
                .function_iter(context)
                .find(|func| &context.functions[func.0].name == "main")
                .expect("Can't find main function!");
            builder
                .compile_function(function)
                .flat_map(|_| builder.finalize())
        }
        Kind::Contract => {
            let mut warnings = Vec::new();
            let mut errors = Vec::new();

            let mut selectors_and_labels: Vec<([u8; 4], Label)> = Vec::new();

            // Compile only the functions which have selectors and gather the selectors and labels.
            for function in module.function_iter(context) {
                if function.has_selector(context) {
                    let selector = function.get_selector(context).unwrap();
                    let label = builder.add_label();
                    check!(
                        builder.compile_function(function),
                        return err(warnings, errors),
                        warnings,
                        errors
                    );
                    selectors_and_labels.push((selector, label));
                }
            }
            let (mut data_section, mut funcs_bytecode, mut reg_seqr) = check!(
                builder.finalize(),
                return err(warnings, errors),
                warnings,
                errors
            );

            let mut bytecode_with_switch =
                build_contract_abi_switch(&mut reg_seqr, &mut data_section, selectors_and_labels);
            bytecode_with_switch.append(&mut funcs_bytecode);
            ok(
                (data_section, bytecode_with_switch, reg_seqr),
                warnings,
                errors,
            )
        }
        Kind::Library | Kind::Predicate => todo!("libraries and predicates coming soon!"),
    }
}

// -------------------------------------------------------------------------------------------------

macro_rules! size_bytes_in_words {
    ($bytes_expr: expr) => {
        ($bytes_expr + 7) / 8
    };
}

// This is a mouthful...
macro_rules! size_bytes_round_up_to_word_alignment {
    ($bytes_expr: expr) => {
        ($bytes_expr + 7) - (($bytes_expr + 7) % 8)
    };
}

struct AsmBuilder<'sc, 'ir> {
    // Data section is used by the rest of code gen to layout const memory.
    data_section: DataSection<'sc>,

    // Register sequencer dishes out new registers and labels.
    reg_seqr: RegisterSequencer,

    // Label map is from IR block to label name.
    label_map: HashMap<Block, Label>,

    // Reg map, const map and var map are all tracking IR values to VM values.  Var map has an
    // optional (None) register until its first assignment.
    reg_map: HashMap<Value, VirtualRegister>,
    ptr_map: HashMap<Pointer, Storage>,

    // Stack base register, copied from $SP at the start, but only if we have stack storage.
    stack_base_reg: Option<VirtualRegister>,

    // The layouts of each aggregate; their whole size in bytes and field offsets in words.
    aggregate_layouts: HashMap<Aggregate, (u64, Vec<FieldLayout>)>,

    // IR context we're compiling.
    context: &'ir Context,

    // Final resulting VM bytecode ops.
    bytecode: Vec<Op<'sc>>,
}

struct FieldLayout {
    offset_in_words: u64, // Use words because LW/SW do.
    size_in_bytes: u64,   // Use bytes because CFEI/MCP do.
}

// NOTE: For stack storage we need to be aware:
// - sizes are in bytes; CFEI reserves in bytes.
// - offsets are in 64-bit words; LW/SW reads/writes to word offsets. XXX Wrap in a WordOffset struct.

#[derive(Clone, Debug)]
pub(super) enum Storage {
    Data(DataId),              // Const storage in the data section.
    Register(VirtualRegister), // Storage in a register.
    Stack(u64), // Storage in the runtime stack starting at an absolute word offset.  Essentially a global.
}

impl<'sc, 'ir> AsmBuilder<'sc, 'ir> {
    fn new(
        data_section: DataSection<'sc>,
        reg_seqr: RegisterSequencer,
        context: &'ir Context,
    ) -> Self {
        AsmBuilder {
            data_section,
            reg_seqr,
            label_map: HashMap::new(),
            reg_map: HashMap::new(),
            ptr_map: HashMap::new(),
            stack_base_reg: None,
            aggregate_layouts: HashMap::new(),
            context,
            bytecode: Vec::new(),
        }
    }

    fn add_locals(&mut self, function: Function) {
        // If they're immutable and have a constant initialiser then they go in the data section.
        // Otherwise they go in runtime allocated space, either a register or on the stack.
        //
        // Stack offsets are in words to both enforce alignment and simplify use with LW/SW.
        let mut stack_base = 0_u64;
        for (_name, ptr) in function.locals_iter(&self.context) {
            let ptr_content = &self.context.pointers[ptr.0];
            if !ptr_content.is_mutable && ptr_content.initializer.is_some() {
                let constant = ptr_content.initializer.as_ref().unwrap();
                let lit = ir_constant_to_ast_literal(constant);
                let data_id = self.data_section.insert_data_value(&lit);
                self.ptr_map.insert(*ptr, Storage::Data(data_id));
            } else {
                match ptr_content.ty {
                    Type::Unit | Type::Bool | Type::Uint(_) => {
                        let reg = self.reg_seqr.next();
                        self.ptr_map.insert(*ptr, Storage::Register(reg));
                    }
                    Type::B256 => {
                        self.ptr_map.insert(*ptr, Storage::Stack(stack_base));
                        stack_base += 4;
                    }
                    Type::String(count) => {
                        self.ptr_map.insert(*ptr, Storage::Stack(stack_base));

                        // XXX `count` is a CHAR count, not BYTE count.  We need to count the size
                        // of the string before allocating.  For now assuming CHAR == BYTE.
                        stack_base += size_bytes_in_words!(count);
                    }
                    Type::Array(aggregate) => {
                        // Store this aggregate at the current stack base.
                        self.ptr_map.insert(*ptr, Storage::Stack(stack_base));

                        // Reserve space by incrementing the base.
                        stack_base += size_bytes_in_words!(self.aggregate_size(&aggregate));
                    }
                    Type::Struct(aggregate) => {
                        // Store this aggregate at the current stack base.
                        self.ptr_map.insert(*ptr, Storage::Stack(stack_base));

                        // Reserve space by incrementing the base.
                        stack_base += size_bytes_in_words!(self.aggregate_size(&aggregate));
                    }
                    Type::Union(aggregate) => {
                        // Store this aggregate AND a 64bit tag at the current stack base.
                        self.ptr_map.insert(*ptr, Storage::Stack(stack_base));

                        // Reserve space by incrementing the base.
                        stack_base +=
                            size_bytes_in_words!(self.aggregate_max_field_size(&aggregate));
                    }
                    Type::ContractCaller(_) => {
                        self.ptr_map.insert(*ptr, Storage::Stack(stack_base));

                        // Reserve space for the contract address only.
                        stack_base += 4;
                    }
                    Type::Contract => {
                        unimplemented!("contract on the stack?")
                    }
                };
            }
        }

        // Reserve space on the stack for ALL our locals which require it.
        if stack_base > 0 {
            let base_reg = self.reg_seqr.next();
            self.bytecode.push(Op::register_move(
                base_reg.clone(),
                VirtualRegister::Constant(ConstantRegister::StackPointer),
                crate::span::Span {
                    span: pest::Span::new(" ", 0, 0).unwrap(),
                    path: None,
                },
            ));
            self.bytecode.push(Op::unowned_stack_allocate_memory(
                VirtualImmediate24::new(
                    stack_base * 8,
                    crate::span::Span {
                        span: pest::Span::new(" ", 0, 0).unwrap(),
                        path: None,
                    },
                )
                .unwrap(),
            ));
            self.stack_base_reg = Some(base_reg);
        }
    }

    fn add_block_label(&mut self, block: Block) {
        if &block.get_label(&self.context) != "entry" {
            let label = self.block_to_label(&block);
            self.bytecode.push(Op::jump_label(
                label,
                crate::span::Span {
                    span: pest::Span::new(" ", 0, 0).unwrap(),
                    path: None,
                },
            ))
        }
    }

    fn add_label(&mut self) -> Label {
        let label = self.reg_seqr.get_label();
        self.bytecode.push(Op::jump_label(
            label.clone(),
            crate::span::Span {
                span: pest::Span::new(" ", 0, 0).unwrap(),
                path: None,
            },
        ));
        label
    }

    fn finalize(self) -> CompileResult<'sc, (DataSection<'sc>, Vec<Op<'sc>>, RegisterSequencer)> {
        ok(
            (self.data_section, self.bytecode, self.reg_seqr),
            Vec::new(),
            Vec::new(),
        )
    }

    fn compile_function(&mut self, function: Function) -> CompileResult<'sc, ()> {
        // Compile instructions.
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        self.add_locals(function);
        for block in function.block_iter(&self.context) {
            self.add_block_label(block);
            for instr_val in block.instruction_iter(&self.context) {
                check!(
                    self.compile_instruction(&block, &instr_val),
                    return err(warnings, errors),
                    warnings,
                    errors
                );
            }
        }
        ok((), warnings, errors)
    }

    fn compile_instruction(&mut self, block: &Block, instr_val: &Value) -> CompileResult<'sc, ()> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        if let ValueContent::Instruction(instruction) = &self.context.values[instr_val.0] {
            match instruction {
                Instruction::AsmBlock(asm, args) => {
                    check!(
                        self.compile_asm_block(instr_val, asm, args),
                        return err(warnings, errors),
                        warnings,
                        errors
                    )
                }
                Instruction::Branch(to_block) => self.compile_branch(block, to_block),
                Instruction::Call(..) => {
                    errors.push(CompileError::Internal(
                        "Calls are not yet supported.",
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    ));
                    return err(warnings, errors);
                }
                Instruction::ConditionalBranch {
                    cond_value,
                    true_block,
                    false_block,
                } => self.compile_conditional_branch(cond_value, block, true_block, false_block),
                Instruction::ExtractElement {
                    array,
                    ty,
                    index_val,
                } => self.compile_extract_element(instr_val, array, ty, index_val),
                Instruction::ExtractValue {
                    aggregate,
                    ty,
                    indices,
                } => self.compile_extract_value(instr_val, aggregate, ty, indices),
                Instruction::GetPointer(ptr) => self.compile_get_pointer(instr_val, ptr),
                Instruction::InsertElement {
                    array,
                    ty,
                    value,
                    index_val,
                } => self.compile_insert_element(instr_val, array, ty, value, index_val),
                Instruction::InsertValue {
                    aggregate,
                    ty,
                    value,
                    indices,
                } => self.compile_insert_value(instr_val, aggregate, ty, value, indices),
                Instruction::Load(ptr) => self.compile_load(instr_val, ptr),
                Instruction::Phi(_) => {
                    // Managing the phi value is done in br and cbr compilation.
                    ()
                }
                Instruction::Ret(ret_val, ty) => self.compile_ret(ret_val, ty),
                Instruction::Store { ptr, stored_val } => self.compile_store(ptr, stored_val),
            }
        } else {
            errors.push(CompileError::Internal(
                "Value not an instruction.",
                crate::span::Span {
                    span: pest::Span::new(" ", 0, 0).unwrap(),
                    path: None,
                },
            ));
        }
        ok((), warnings, errors)
    }

    // OK, I began by trying to translate the IR ASM block data structures back into AST data
    // structures which I could feed to the code in asm_generation/expression/mod.rs where it
    // compiles the inline ASM.  But it's more work to do that than to just re-implement that
    // algorithm with the IR data here.

    fn compile_asm_block(
        &mut self,
        instr_val: &Value,
        asm: &AsmBlock,
        asm_args: &[AsmArg],
    ) -> CompileResult<'sc, ()> {
        let mut warnings: Vec<CompileWarning<'sc>> = Vec::new();
        let mut errors: Vec<CompileError<'sc>> = Vec::new();
        let mut inline_reg_map = HashMap::new();
        let mut inline_ops = Vec::new();
        for AsmArg { name, initializer } in asm_args {
            assert_or_warn!(
                ConstantRegister::parse_register_name(name).is_none(),
                warnings,
                crate::span::Span {
                    span: pest::Span::new(" ", 0, 0).unwrap(),
                    path: None,
                },
                Warning::ShadowingReservedRegister {
                    reg_name: name.into()
                }
            );
            let arg_reg = initializer
                .map(|init_val| self.value_to_register(&init_val))
                .unwrap_or_else(|| self.reg_seqr.next());
            inline_reg_map.insert(name, arg_reg);
        }

        let realize_register = |reg_name: &String| {
            inline_reg_map.get(reg_name).cloned().or_else(|| {
                ConstantRegister::parse_register_name(reg_name)
                    .map(|reg| VirtualRegister::Constant(reg))
            })
        };

        // For each opcode in the asm expression, attempt to parse it into an opcode and
        // replace references to the above registers with the newly allocated ones.
        let asm_block = &self.context.asm_blocks[asm.0];
        for op in &asm_block.body {
            let replaced_registers = op
                .args
                .iter()
                .map(|reg_name| -> Result<_, CompileError> {
                    realize_register(reg_name).ok_or_else(|| CompileError::UnknownRegister {
                        span: crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                        initialized_registers: inline_reg_map
                            .iter()
                            .map(|(name, _)| (*name).clone())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    })
                })
                .filter_map(|res| match res {
                    Err(e) => {
                        errors.push(e);
                        None
                    }
                    Ok(o) => Some(o),
                })
                .collect::<Vec<VirtualRegister>>();

            // Parse the actual op and registers.
            // XXX TODO: This 'unchecked' version is here because it was too hard to resolve
            // borrowing the errors from `Op::parse_opcode()` as 'sc.  This needs to be fixed
            // somehow, maybe with just static spans (COMING SOON!) or maybe static errors.
            let opcode = Op::parse_opcode_unchecked(
                &op.name,
                replaced_registers,
                op.immediate.as_ref().map(|imm_str| Ident {
                    primary_name: imm_str,
                    span: crate::span::Span {
                        span: pest::Span::new(" ", 0, 0).unwrap(),
                        path: None,
                    },
                }),
            );

            inline_ops.push(Op {
                opcode: either::Either::Left(opcode),
                comment: String::new(),
                owning_span: None,
            });
        }

        // Now, load the designated asm return register into the desired return register, but only
        // if it was named.
        if let Some(ret_reg_name) = &asm_block.return_name {
            // Lookup and replace the return register.
            let ret_reg = match realize_register(&ret_reg_name) {
                Some(reg) => reg,
                None => {
                    errors.push(CompileError::UnknownRegister {
                        span: crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                        initialized_registers: inline_reg_map
                            .iter()
                            .map(|(name, _)| name.to_string())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    });
                    return err(warnings, errors);
                }
            };
            let instr_reg = self.reg_seqr.next();
            inline_ops.push(Op::unowned_register_move_comment(
                instr_reg.clone(),
                ret_reg.clone(),
                "return value from inline asm",
            ));
            self.reg_map.insert(*instr_val, instr_reg);
        }

        self.bytecode.append(&mut inline_ops);

        ok((), warnings, errors)
    }

    fn compile_branch(&mut self, from_block: &Block, to_block: &Block) {
        self.compile_branch_to_phi_value(from_block, to_block);

        let label = self.block_to_label(to_block);
        self.bytecode.push(Op::jump_to_label(label));
    }

    fn compile_conditional_branch(
        &mut self,
        cond_value: &Value,
        from_block: &Block,
        true_block: &Block,
        false_block: &Block,
    ) {
        self.compile_branch_to_phi_value(from_block, true_block);
        self.compile_branch_to_phi_value(from_block, false_block);

        let cond_reg = self.value_to_register(cond_value);

        let false_label = self.block_to_label(false_block);
        self.bytecode.push(Op::jump_if_not_equal(
            cond_reg,
            VirtualRegister::Constant(ConstantRegister::One),
            false_label,
        ));

        let true_label = self.block_to_label(true_block);
        self.bytecode.push(Op::jump_to_label(true_label));
    }

    fn compile_branch_to_phi_value(&mut self, from_block: &Block, to_block: &Block) {
        if let Some(local_val) = to_block.get_phi_val_coming_from(&self.context, from_block) {
            let local_reg = self.value_to_register(&local_val);
            let phi_reg = self.value_to_register(&to_block.get_phi(&self.context));
            self.bytecode.push(Op::register_move(
                phi_reg,
                local_reg,
                crate::span::Span {
                    span: pest::Span::new(" ", 0, 0).unwrap(),
                    path: None,
                },
            ));
        }
    }

    fn compile_extract_element(
        &mut self,
        instr_val: &Value,
        array: &Value,
        ty: &Aggregate,
        index_val: &Value,
    ) {
        // Base register should pointer to some stack allocated memory.
        let base_reg = self.value_to_register(array);

        // Index value is the array element index, not byte nor word offset.
        let index_reg = self.value_to_register(index_val);

        // We could put the OOB check here, though I'm now thinking it would be too wasteful.
        // See compile_bounds_assertion() in expression/array.rs (or look in Git history).

        let instr_reg = self.reg_seqr.next();
        let elem_size = self.ir_type_size_in_bytes(&ty.get_elem_type(self.context).unwrap());
        if elem_size <= 8 {
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::MULI(
                    index_reg.clone(),
                    index_reg.clone(),
                    VirtualImmediate12 { value: 8 },
                )),
                comment: "extract_element relative offset".into(),
                owning_span: None,
            });
            let elem_offs_reg = self.reg_seqr.next();
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::ADD(
                    elem_offs_reg.clone(),
                    base_reg.clone(),
                    index_reg,
                )),
                comment: "extract_element absolute offset".into(),
                owning_span: None,
            });
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::LW(
                    instr_reg.clone(),
                    elem_offs_reg,
                    VirtualImmediate12 { value: 0 },
                )),
                comment: "extract_element".into(),
                owning_span: None,
            });
        } else {
            // Value too big for a register, so we return the memory offset.
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::MULI(
                    instr_reg.clone(),
                    index_reg,
                    VirtualImmediate12::new(
                        elem_size,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    )
                    .unwrap(),
                )),
                comment: "extract_element relative offset".into(),
                owning_span: None,
            });
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::ADD(
                    instr_reg.clone(),
                    base_reg.clone(),
                    instr_reg.clone(),
                )),
                comment: "extract_element absolute offset".into(),
                owning_span: None,
            });
        }

        self.reg_map.insert(*instr_val, instr_reg);
    }

    fn compile_extract_value(
        &mut self,
        instr_val: &Value,
        aggregate: &Value,
        ty: &Aggregate,
        indices: &[u64],
    ) {
        // Base register should pointer to some stack allocated memory.
        let base_reg = self.value_to_register(aggregate);
        let (extract_offset, value_size) = self.aggregate_idcs_to_field_layout(ty, indices);

        let instr_reg = self.reg_seqr.next();
        if value_size <= 8 {
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::LW(
                    instr_reg.clone(),
                    base_reg,
                    VirtualImmediate12::new(
                        extract_offset,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    )
                    .unwrap(),
                )),
                comment: format!(
                    "extract_value @ {}",
                    indices
                        .iter()
                        .map(|idx| format!("{}", idx))
                        .collect::<Vec<String>>()
                        .join(",")
                ),
                owning_span: None,
            });
        } else {
            // Value too big for a register, so we return the memory offset.
            self.bytecode.push(Op {
                opcode: either::Either::Left(VirtualOp::ADDI(
                    instr_reg.clone(),
                    base_reg,
                    VirtualImmediate12::new(
                        extract_offset * 8,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    )
                    .unwrap(),
                )),
                comment: "extract address".into(),
                owning_span: None,
            });
        }

        self.reg_map.insert(*instr_val, instr_reg);
    }

    fn compile_get_pointer(&mut self, instr_val: &Value, ptr: &Pointer) {
        // `get_ptr` is like a `load` except the value isn't dereferenced.
        match self.ptr_map.get(ptr) {
            None => unimplemented!("BUG? Uninitialised pointer."),
            Some(storage) => match storage {
                Storage::Data(_data_id) => {
                    // Not sure if we'll ever need this.
                    unimplemented!("TODO get_ptr() into the data section.");
                }
                Storage::Register(var_reg) => {
                    self.reg_map.insert(*instr_val, var_reg.clone());
                }
                Storage::Stack(word_offs) => {
                    let instr_reg = self.reg_seqr.next();
                    self.bytecode.push(Op {
                        opcode: either::Either::Left(VirtualOp::ADDI(
                            instr_reg.clone(),
                            self.stack_base_reg.as_ref().unwrap().clone(),
                            VirtualImmediate12::new(
                                *word_offs * 8,
                                crate::span::Span {
                                    span: pest::Span::new(" ", 0, 0).unwrap(),
                                    path: None,
                                },
                            )
                            .unwrap(),
                        )),
                        comment: "get_ptr".into(),
                        owning_span: None,
                    });
                    self.reg_map.insert(*instr_val, instr_reg);
                }
            },
        }
    }

    fn compile_insert_element(
        &mut self,
        instr_val: &Value,
        array: &Value,
        ty: &Aggregate,
        value: &Value,
        index_val: &Value,
    ) {
        // Base register should point to some stack allocated memory.
        let base_reg = self.value_to_register(array);
        let insert_reg = self.value_to_register(value);

        // Index value is the array element index, not byte nor word offset.
        let index_reg = self.value_to_register(index_val);

        let elem_size = self.ir_type_size_in_bytes(&ty.get_elem_type(self.context).unwrap());
        if elem_size <= 8 {
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::MULI(
                    index_reg.clone(),
                    index_reg.clone(),
                    VirtualImmediate12 { value: 8 },
                )),
                comment: "insert_element relative offset".into(),
                owning_span: None,
            });
            let elem_offs_reg = self.reg_seqr.next();
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::ADD(
                    elem_offs_reg.clone(),
                    base_reg.clone(),
                    index_reg,
                )),
                comment: "insert_element absolute offset".into(),
                owning_span: None,
            });
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::SW(
                    elem_offs_reg,
                    insert_reg,
                    VirtualImmediate12 { value: 0 },
                )),
                comment: "insert_element".into(),
                owning_span: None,
            });
        } else {
            // Element size is larger than 8; we switch to bytewise offsets and sizes and use MCP.
            let elem_index_offs_reg = self.reg_seqr.next();
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::MULI(
                    elem_index_offs_reg.clone(),
                    index_reg,
                    VirtualImmediate12::new(
                        elem_size,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    )
                    .unwrap(),
                )),
                comment: "insert_element relative offset".into(),
                owning_span: None,
            });
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::ADD(
                    elem_index_offs_reg.clone(),
                    base_reg.clone(),
                    elem_index_offs_reg.clone(),
                )),
                comment: "insert_element absolute offset".into(),
                owning_span: None,
            });
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::MCPI(
                    elem_index_offs_reg,
                    insert_reg,
                    VirtualImmediate12::new(
                        elem_size,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    )
                    .unwrap(),
                )),
                comment: "insert_element store value".into(),
                owning_span: None,
            });
        }

        // We set the 'instruction' register to the base register, so that cascading inserts will
        // work.
        self.reg_map.insert(*instr_val, base_reg);
    }

    fn compile_insert_value(
        &mut self,
        instr_val: &Value,
        aggregate: &Value,
        ty: &Aggregate,
        value: &Value,
        indices: &[u64],
    ) {
        // Base register should point to some stack allocated memory.
        let base_reg = self.value_to_register(aggregate);

        let insert_reg = self.value_to_register(value);
        let (insert_offs, value_size) = self.aggregate_idcs_to_field_layout(ty, indices);

        if value_size <= 8 {
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::SW(
                    base_reg.clone(),
                    insert_reg,
                    VirtualImmediate12::new(
                        insert_offs,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    )
                    .unwrap(),
                )),
                comment: format!(
                    "insert_value @ {}",
                    indices
                        .iter()
                        .map(|idx| format!("{}", idx))
                        .collect::<Vec<String>>()
                        .join(",")
                ),
                owning_span: None,
            });
        } else {
            let offs_reg = self.reg_seqr.next();
            self.bytecode.push(Op {
                opcode: either::Either::Left(VirtualOp::ADDI(
                    offs_reg.clone(),
                    base_reg.clone(),
                    VirtualImmediate12::new(
                        insert_offs * 8,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    )
                    .unwrap(),
                )),
                comment: "insert_value get offset".into(),
                owning_span: None,
            });
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::MCPI(
                    offs_reg,
                    insert_reg,
                    VirtualImmediate12::new(
                        value_size,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    )
                    .unwrap(),
                )),
                comment: "insert_value store value".into(),
                owning_span: None,
            });
        }

        // We set the 'instruction' register to the base register, so that cascading inserts will
        // work.
        self.reg_map.insert(*instr_val, base_reg);
    }

    fn compile_load(&mut self, instr_val: &Value, ptr: &Pointer) {
        let load_size_in_words =
            size_bytes_in_words!(self.ir_type_size_in_bytes(ptr.get_type(self.context)));
        let instr_reg = self.reg_seqr.next();
        match self.ptr_map.get(ptr) {
            None => unimplemented!("BUG? Uninitialised pointer."),
            Some(storage) => match storage {
                Storage::Data(data_id) => {
                    self.bytecode.push(Op::unowned_load_data_comment(
                        instr_reg.clone(),
                        data_id.clone(),
                        "load constant",
                    ));
                }
                Storage::Register(var_reg) => {
                    self.bytecode.push(Op::register_move(
                        instr_reg.clone(),
                        var_reg.clone(),
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    ));
                }
                Storage::Stack(word_offs) => {
                    // XXX Need to check for zero sized types?
                    if load_size_in_words == 1 {
                        // Value can fit in a register, so we load the value.
                        self.bytecode.push(Op {
                            opcode: Either::Left(VirtualOp::LW(
                                instr_reg.clone(),
                                self.stack_base_reg.as_ref().unwrap().clone(),
                                VirtualImmediate12::new(
                                    *word_offs,
                                    crate::span::Span {
                                        span: pest::Span::new(" ", 0, 0).unwrap(),
                                        path: None,
                                    },
                                )
                                .unwrap(),
                            )),
                            comment: "load value".into(),
                            owning_span: None,
                        });
                    } else {
                        // Value too big for a register, so we return the memory offset.  This is
                        // what LW to the data section does, via LWDataId.
                        self.bytecode.push(Op {
                            opcode: either::Either::Left(VirtualOp::ADDI(
                                instr_reg.clone(),
                                self.stack_base_reg.as_ref().unwrap().clone(),
                                VirtualImmediate12::new(
                                    word_offs * 8,
                                    crate::span::Span {
                                        span: pest::Span::new(" ", 0, 0).unwrap(),
                                        path: None,
                                    },
                                )
                                .unwrap(),
                            )),
                            comment: "load address".into(),
                            owning_span: None,
                        });
                    }
                }
            },
        }
        self.reg_map.insert(*instr_val, instr_reg);
    }

    // XXX This is copied from ret_or_retd_value() above, streamlined for IR types.
    fn compile_ret(&mut self, ret_val: &Value, ret_type: &Type) {
        if ret_type == &Type::Unit {
            // Unit returns should always be zero, although because they can be omitted from
            // functions, the register is sometimes uninitialized. Manually return zero in this
            // case.
            self.bytecode.push(Op {
                opcode: Either::Left(VirtualOp::RET(VirtualRegister::Constant(
                    ConstantRegister::Zero,
                ))),
                owning_span: None,
                comment: "returning unit as zero".into(),
            });
        } else {
            let ret_reg = self.value_to_register(ret_val);
            let size_in_bytes = self.ir_type_size_in_bytes(ret_type);

            if size_in_bytes <= 8 {
                self.bytecode.push(Op {
                    owning_span: None,
                    opcode: Either::Left(VirtualOp::RET(ret_reg)),
                    comment: "".into(),
                });
            } else {
                // If the type is larger than one word, then we use RETD to return data.  First put
                // the size into the data section, then add a LW to get it, then add a RETD which
                // uses it.
                let size_reg = self.reg_seqr.next();
                let size_data_id = self
                    .data_section
                    .insert_data_value(&Literal::U64(size_in_bytes));

                self.bytecode.push(Op {
                    opcode: Either::Left(VirtualOp::LWDataId(size_reg.clone(), size_data_id)),
                    owning_span: None,
                    comment: "loading size for RETD".into(),
                });
                self.bytecode.push(Op {
                    owning_span: None,
                    opcode: Either::Left(VirtualOp::RETD(ret_reg, size_reg)),
                    comment: "".into(),
                });
            }
        }
    }

    fn compile_store(&mut self, ptr: &Pointer, stored_val: &Value) {
        let stored_reg = self.value_to_register(stored_val);
        let is_struct_ptr = ptr.is_struct_ptr(self.context);
        match self.ptr_map.get(ptr) {
            None => unreachable!("Bug! Trying to store to an unknown pointer."),
            Some(storage) => match storage {
                Storage::Data(_) => unreachable!("BUG! Trying to store to the data section."),
                Storage::Register(reg) => {
                    self.bytecode.push(Op::register_move(
                        reg.clone(),
                        stored_reg,
                        crate::span::Span {
                            span: pest::Span::new(" ", 0, 0).unwrap(),
                            path: None,
                        },
                    ));
                }
                Storage::Stack(word_offs) => {
                    let word_offs = *word_offs;
                    let store_size_in_words = size_bytes_in_words!(
                        self.ir_type_size_in_bytes(ptr.get_type(self.context))
                    );
                    match store_size_in_words {
                        // We can have empty sized types which we can ignore.
                        0 => (),
                        1 => {
                            // A single word can be stored with SW.
                            let stored_reg = if !is_struct_ptr {
                                // stored_reg is a value.
                                stored_reg
                            } else {
                                // stored_reg is a pointer, even though size is 1.  We need to load it.
                                let tmp_reg = self.reg_seqr.next();
                                self.bytecode.push(Op {
                                    opcode: Either::Left(VirtualOp::LW(
                                        tmp_reg.clone(),
                                        stored_reg,
                                        VirtualImmediate12 { value: 0 },
                                    )),
                                    comment: "load for store".into(),
                                    owning_span: None,
                                });
                                tmp_reg
                            };
                            self.bytecode.push(Op {
                                opcode: Either::Left(VirtualOp::SW(
                                    self.stack_base_reg.as_ref().unwrap().clone(),
                                    stored_reg,
                                    VirtualImmediate12::new(
                                        word_offs,
                                        crate::span::Span {
                                            span: pest::Span::new(" ", 0, 0).unwrap(),
                                            path: None,
                                        },
                                    )
                                    .unwrap(),
                                )),
                                comment: "store value".into(),
                                owning_span: None,
                            });
                        }
                        _ => {
                            // Bigger than 1 word needs a MCPI.  XXX Or MCP if it's huge.
                            let dest_reg = self.reg_seqr.next();
                            self.bytecode.push(Op {
                                opcode: either::Either::Left(VirtualOp::ADDI(
                                    dest_reg.clone(),
                                    self.stack_base_reg.as_ref().unwrap().clone(),
                                    VirtualImmediate12::new(
                                        word_offs * 8,
                                        crate::span::Span {
                                            span: pest::Span::new(" ", 0, 0).unwrap(),
                                            path: None,
                                        },
                                    )
                                    .unwrap(),
                                )),
                                comment: "store get offset".into(),
                                owning_span: None,
                            });

                            self.bytecode.push(Op {
                                opcode: Either::Left(VirtualOp::MCPI(
                                    dest_reg,
                                    stored_reg,
                                    VirtualImmediate12::new(
                                        store_size_in_words * 8,
                                        crate::span::Span {
                                            span: pest::Span::new(" ", 0, 0).unwrap(),
                                            path: None,
                                        },
                                    )
                                    .unwrap(),
                                )),
                                comment: "store value".into(),
                                owning_span: None,
                            });
                        }
                    }
                }
            },
        };
    }

    fn value_to_register(&mut self, value: &Value) -> VirtualRegister {
        match self.reg_map.get(value) {
            Some(reg) => reg.clone(),
            None => {
                match &self.context.values[value.0] {
                    // Handle constants.
                    ValueContent::Constant(constant) => {
                        match &constant.value {
                            ConstantValue::Struct(_) | ConstantValue::Array(_) => {
                                // A constant struct or array.  We still allocate space for it on
                                // the stack, but create the field or element initialisers
                                // recursively.

                                // Get the total size.
                                let total_size = size_bytes_round_up_to_word_alignment!(
                                    self.constant_size_in_bytes(constant)
                                );

                                let start_reg = self.reg_seqr.next();

                                // We can have zero sized structs and maybe arrays?
                                if total_size > 0 {
                                    // Save the stack pointer.
                                    self.bytecode.push(Op::register_move(
                                        start_reg.clone(),
                                        VirtualRegister::Constant(ConstantRegister::StackPointer),
                                        crate::span::Span {
                                            span: pest::Span::new(" ", 0, 0).unwrap(),
                                            path: None,
                                        },
                                    ));

                                    self.bytecode.push(Op::unowned_stack_allocate_memory(
                                        VirtualImmediate24::new(
                                            total_size,
                                            crate::span::Span {
                                                span: pest::Span::new(" ", 0, 0).unwrap(),
                                                path: None,
                                            },
                                        )
                                        .unwrap(),
                                    ));

                                    // Fill in the fields.
                                    self.initialise_constant_memory(constant, &start_reg, 0);
                                }

                                // Return the start ptr.
                                start_reg
                            }

                            ConstantValue::Undef
                            | ConstantValue::Unit
                            | ConstantValue::Bool(_)
                            | ConstantValue::Uint(_)
                            | ConstantValue::B256(_)
                            | ConstantValue::String(_) => {
                                // Get the constant into the namespace.
                                let lit = ir_constant_to_ast_literal(constant);
                                let data_id = self.data_section.insert_data_value(&lit);

                                // Allocate a register for it, and a load instruction.
                                let reg = self.reg_seqr.next();
                                self.bytecode.push(Op {
                                    opcode: either::Either::Left(VirtualOp::LWDataId(
                                        reg.clone(),
                                        data_id,
                                    )),
                                    comment: "literal instantiation".into(),
                                    owning_span: None,
                                });

                                // Insert the value into the map.
                                self.reg_map.insert(*value, reg.clone());

                                // Return register.
                                reg
                            }
                        }
                    }

                    _otherwise => {
                        // Just make a new register for this value.
                        let reg = self.reg_seqr.next();
                        self.reg_map.insert(*value, reg.clone());
                        reg
                    }
                }
            }
        }
    }

    fn constant_size_in_bytes(&mut self, constant: &Constant) -> u64 {
        match &constant.value {
            ConstantValue::Undef => self.ir_type_size_in_bytes(&constant.ty),
            ConstantValue::Unit => 8,
            ConstantValue::Bool(_) => 8,
            ConstantValue::Uint(_) => 8,
            ConstantValue::B256(_) => 32,
            ConstantValue::String(s) => s.len() as u64, // String::len() returns the byte size, not char count.
            ConstantValue::Array(elems) => {
                if elems.is_empty() {
                    0
                } else {
                    self.constant_size_in_bytes(&elems[0]) * elems.len() as u64
                }
            }
            ConstantValue::Struct(fields) => fields
                .iter()
                .fold(0, |acc, field| acc + self.constant_size_in_bytes(field)),
        }
    }

    fn initialise_constant_memory(
        &mut self,
        constant: &Constant,
        start_reg: &VirtualRegister,
        offs_in_words: u64,
    ) -> u64 {
        match &constant.value {
            ConstantValue::Undef => {
                // We don't need to actually create an initialiser, but we do need to return the
                // field size in words.
                size_bytes_in_words!(self.ir_type_size_in_bytes(&constant.ty))
            }
            ConstantValue::Unit
            | ConstantValue::Bool(_)
            | ConstantValue::Uint(_)
            | ConstantValue::B256(_)
            | ConstantValue::String(_) => {
                // Get the constant into the namespace.
                let lit = ir_constant_to_ast_literal(constant);
                let data_id = self.data_section.insert_data_value(&lit);

                // Load the initialiser value.
                let init_reg = self.reg_seqr.next();
                self.bytecode.push(Op {
                    opcode: either::Either::Left(VirtualOp::LWDataId(init_reg.clone(), data_id)),
                    comment: "literal instantiation for aggregate field".into(),
                    owning_span: None,
                });

                // Write the initialiser to memory.
                self.bytecode.push(Op {
                    opcode: Either::Left(VirtualOp::SW(
                        start_reg.clone(),
                        init_reg,
                        VirtualImmediate12::new(
                            offs_in_words,
                            crate::span::Span {
                                span: pest::Span::new(" ", 0, 0).unwrap(),
                                path: None,
                            },
                        )
                        .unwrap(),
                    )),
                    comment: format!(
                        "initialise aggregate field at stack offset {}",
                        offs_in_words
                    ),
                    owning_span: None,
                });

                // Return the constant size in words.
                match &constant.value {
                    ConstantValue::B256(_) => 4,
                    ConstantValue::String(s) => size_bytes_in_words!(s.len() as u64),
                    _otherwise => 1,
                }
            }

            ConstantValue::Array(items) | ConstantValue::Struct(items) => {
                let mut cur_offs = offs_in_words;
                for item in items {
                    let item_size = self.initialise_constant_memory(item, start_reg, cur_offs);
                    cur_offs += item_size;
                }
                cur_offs
            }
        }
    }

    fn block_to_label(&mut self, block: &Block) -> Label {
        match self.label_map.get(block) {
            Some(label) => label.clone(),
            None => {
                let label = self.reg_seqr.get_label();
                self.label_map.insert(*block, label.clone());
                label
            }
        }
    }

    // Aggregate size in bytes.
    fn aggregate_size(&mut self, aggregate: &Aggregate) -> u64 {
        self.analyze_aggregate(aggregate);
        self.aggregate_layouts.get(aggregate).unwrap().0
    }

    // Size of largest aggregate field in bytes.
    fn aggregate_max_field_size(&mut self, aggregate: &Aggregate) -> u64 {
        self.analyze_aggregate(aggregate);
        self.aggregate_layouts
            .get(aggregate)
            .unwrap()
            .1
            .iter()
            .map(|layout| layout.size_in_bytes)
            .max()
            .unwrap_or(0)
    }

    // Aggregate (nested) field offset in words and size in bytes.
    fn aggregate_idcs_to_field_layout(
        &mut self,
        aggregate: &Aggregate,
        idcs: &[u64],
    ) -> (u64, u64) {
        self.analyze_aggregate(aggregate);

        idcs.iter()
            .fold(
                ((0, 0), Type::Struct(*aggregate)),
                |((offs, _), ty), idx| match ty {
                    Type::Struct(aggregate) => {
                        let agg_content = &self.context.aggregates[aggregate.0];
                        let field_type = agg_content.field_types()[*idx as usize];

                        let field_layout =
                            &self.aggregate_layouts.get(&aggregate).unwrap().1[*idx as usize];

                        (
                            (
                                offs + field_layout.offset_in_words,
                                field_layout.size_in_bytes,
                            ),
                            field_type,
                        )
                    }
                    _otherwise => panic!("Attempt to access field in non-aggregate."),
                },
            )
            .0
    }

    fn analyze_aggregate(&mut self, aggregate: &Aggregate) {
        if self.aggregate_layouts.contains_key(aggregate) {
            return;
        }

        match &self.context.aggregates[aggregate.0] {
            AggregateContent::FieldTypes(field_types) => {
                let (total_in_words, offsets) =
                    field_types
                        .iter()
                        .fold((0, Vec::new()), |(cur_offset, mut layouts), ty| {
                            let field_size_in_bytes = self.ir_type_size_in_bytes(ty);
                            layouts.push(FieldLayout {
                                offset_in_words: cur_offset,
                                size_in_bytes: field_size_in_bytes,
                            });
                            (
                                cur_offset + size_bytes_in_words!(field_size_in_bytes),
                                layouts,
                            )
                        });
                self.aggregate_layouts
                    .insert(*aggregate, (total_in_words * 8, offsets));
            }
            AggregateContent::ArrayType(el_type, count) => {
                // Careful!  We *could* wrap the aggregate in Type::Array and call
                // ir_type_size_in_bytes() BUT we'd then enter a recursive loop.
                let el_size = self.ir_type_size_in_bytes(el_type);
                self.aggregate_layouts
                    .insert(*aggregate, (count * el_size, Vec::new()));
            }
        }
    }

    fn ir_type_size_in_bytes(&mut self, ty: &Type) -> u64 {
        match ty {
            Type::Unit | Type::Bool | Type::Uint(_) => 8,
            Type::B256 => 32,
            Type::String(n) => *n,
            Type::Array(aggregate) | Type::Struct(aggregate) => {
                self.analyze_aggregate(aggregate);
                self.aggregate_size(aggregate)
            }
            Type::Union(aggregate) => {
                self.analyze_aggregate(aggregate);
                self.aggregate_max_field_size(aggregate)
            }
            Type::ContractCaller(_) => {
                // We only store the address.
                32
            }
            Type::Contract => {
                unimplemented!("do contract/contract caller have/need a size?")
            }
        }
    }
}

fn ir_constant_to_ast_literal<'sc>(constant: &Constant) -> Literal<'sc> {
    match &constant.value {
        ConstantValue::Undef => unreachable!("Cannot convert 'undef' to a literal."),
        ConstantValue::Unit => Literal::U64(0), // No unit.
        ConstantValue::Bool(b) => Literal::Boolean(*b),
        ConstantValue::Uint(n) => Literal::U64(*n),
        ConstantValue::B256(bs) => Literal::B256(bs.clone()),
        ConstantValue::String(_) => {
            Literal::String("STRINGS ARE UNIMPLEMENTED UNTIL WE CAN GET AROUND 'sc")
        }
        ConstantValue::Array(_) => unimplemented!(),
        ConstantValue::Struct(_) => unimplemented!(),
    }
}

// -------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ir::parser::parse;

    fn simple_test(input: &str, expected: &str) {
        let ir = parse(input).expect("parsed ir");
        let asm_result = compile_ir_to_asm(
            &ir,
            &BuildConfig {
                file_name: std::sync::Arc::new("".into()),
                dir_of_code: std::sync::Arc::new("".into()),
                manifest_path: std::sync::Arc::new("".into()),
                print_intermediate_asm: false,
                print_finalized_asm: false,
                print_ir: false,
            },
        );

        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let asm = asm_result.unwrap(&mut warnings, &mut errors);
        assert!(warnings.is_empty() && errors.is_empty());

        let asm_script = format!("{}", asm);
        if asm_script != expected {
            println!("{}", prettydiff::diff_lines(expected, &asm_script));
            assert!(false);
        }
    }

    #[test]
    fn impl_ret_int() {
        let input = r#"script script {
    fn main() -> u64 {
        entry:
        v0 = const u64 42
        ret u64 v0
    }
}
"#;

        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
lw   $r0 data_0               ; literal instantiation
ret  $r0
noop                          ; word-alignment of data section
.data:
data_0 .u64 0x2a
"#;

        simple_test(input, expected);
    }

    #[test]
    fn if_expr() {
        let input = r#"script script {
    fn main() -> u64 {
        entry:
        v0 = const bool false
        cbr v0, block0, block1

        block0:
        v1 = const u64 1000000
        br block2

        block1:
        v2 = const u64 42
        br block2

        block2:
        v3 = phi(block0: v1, block1: v2)
        ret u64 v3
    }
}
"#;
        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
lw   $r0 data_0               ; literal instantiation
jnei $r0 $one i11
lw   $r0 data_1               ; literal instantiation
move $r1 $r0
ji   i13
lw   $r0 data_2               ; literal instantiation
move $r1 $r0
ret  $r1
noop                          ; word-alignment of data section
.data:
data_0 .bool 0x00
data_1 .u64 0xf4240
data_2 .u64 0x2a
"#;

        simple_test(input, expected);
    }

    #[test]
    fn let_reassign_while_loop() {
        let input = r#"script script {
    fn main() -> bool {
        local ptr bool a

        entry:
        v0 = const bool true
        store v0, ptr bool a
        br while

        while:
        v1 = load ptr bool a
        cbr v1, while_body, end_while

        while_body:
        v2 = load ptr bool a
        cbr v2, block0, block1

        block0:
        v3 = phi(while_body: v2)
        v4 = const bool false
        br block1

        block1:
        v5 = phi(while_body: v2, block0: v4)
        store v5, ptr bool a
        br while

        end_while:
        v6 = load ptr bool a
        ret bool v6
    }
}
"#;
        // THIS IS EQIVALENT TO THE ORIGINAL.  Except for a couple of redundant MOVEs.  This is
        // because we're just putting locals into registers rather than mutable storage
        // (Load/Store) and the IR doesn't optimise for redundancies yet.
        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
lw   $r0 data_0               ; literal instantiation
move $r1 $r0
move $r0 $r1
jnei $r0 $one i18
move $r0 $r1
move $r2 $r0
move $r2 $r0
jnei $r0 $one i16
lw   $r0 data_1               ; literal instantiation
move $r2 $r0
move $r1 $r2
ji   i8
move $r0 $r1
ret  $r0
noop                          ; word-alignment of data section
.data:
data_0 .bool 0x01
data_1 .bool 0x00
"#;
        simple_test(input, expected);
    }

    #[test]
    fn tiny_asm_block() {
        let input = r#"script script {
    fn main() -> u64 {
        entry:
        v0 = asm(r1) -> r1 {
            bhei   r1
        }
        ret u64 v0
    }
}
"#;

        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
bhei $r0
move $r1 $r0                  ; return value from inline asm
ret  $r1
.data:
"#;
        simple_test(input, expected);
    }

    #[test]
    fn bigger_asm_block() {
        let input = r#"script script {
    fn main() -> bool {
        local ptr b256 a

        entry:
        v0 = const b256 0x0202020202020202020202020202020202020202020202020202020202020202
        store v0, ptr b256 a
        v1 = load ptr b256 a
        v2 = const b256 0x0303030303030303030303030303030303030303030303030303030303030303
        v3 = asm(lhs: v1, rhs: v2, sz, res) -> res {
            addi   sz zero i32
            meq    res lhs rhs sz
        }
        br block0

        block0:
        v4 = phi(entry: v3)
        ret bool v4
    }
}
"#;

        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
move $r0 $sp
cfei i32
lw   $r1 data_0               ; literal instantiation
addi $r2 $r0 i0               ; store get offset
mcpi $r2 $r1 i32              ; store value
addi $r1 $r0 i0               ; load address
lw   $r0 data_1               ; literal instantiation
addi $r2 $zero i32
meq  $r3 $r1 $r0 $r2
move $r0 $r3                  ; return value from inline asm
move $r1 $r0
ret  $r1
noop                          ; word-alignment of data section
.data:
data_0 .b256 0x0202020202020202020202020202020202020202020202020202020202020202
data_1 .b256 0x0303030303030303030303030303030303030303030303030303030303030303
"#;
        simple_test(input, expected);
    }

    #[test]
    fn simple_struct() {
        let input = r#"script script {
    fn main() -> u64 {
        local ptr { u64, u64 } record

        entry:
        v0 = const { u64, u64 } { u64 undef, u64 undef }
        v1 = const u64 40
        v2 = insert_value v0, { u64, u64 }, v1, 0
        v3 = const u64 2
        v4 = insert_value v2, { u64, u64 }, v3, 1
        store v4, ptr { u64, u64 } record
        v5 = get_ptr ptr { u64, u64 } record
        v6 = extract_value v5, { u64, u64 }, 0
        ret u64 v6
    }

}
"#;

        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
move $r0 $sp
cfei i16
move $r1 $sp
cfei i16
lw   $r2 data_0               ; literal instantiation
sw   $r1 $r2 i0               ; insert_value @ 0
lw   $r2 data_1               ; literal instantiation
sw   $r1 $r2 i1               ; insert_value @ 1
addi $r2 $r0 i0               ; store get offset
mcpi $r2 $r1 i16              ; store value
addi $r1 $r0 i0               ; get_ptr
lw   $r0 $r1 i0               ; extract_value @ 0
ret  $r0
.data:
data_0 .u64 0x28
data_1 .u64 0x02
"#;
        simple_test(input, expected);
    }

    #[test]
    fn mutable_struct() {
        let input = r#"script script {
    fn main() -> u64 {
        local mut ptr { u64, u64 } record

        entry:
        v0 = const { u64, u64 } { u64 undef, u64 undef }
        v1 = const u64 40
        v2 = insert_value v0, { u64, u64 }, v1, 0
        v3 = const u64 2
        v4 = insert_value v2, { u64, u64 }, v3, 1
        store v4, mut ptr { u64, u64 } record
        v5 = get_ptr mut ptr { u64, u64 } record
        v6 = const u64 50
        v7 = insert_value v5, { u64, u64 }, v6, 0
        v8 = get_ptr mut ptr { u64, u64 } record
        v9 = extract_value v8, { u64, u64 }, 1
        ret u64 v9

    }
}
"#;

        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
move $r0 $sp
cfei i16
move $r1 $sp
cfei i16
lw   $r2 data_0               ; literal instantiation
sw   $r1 $r2 i0               ; insert_value @ 0
lw   $r2 data_1               ; literal instantiation
sw   $r1 $r2 i1               ; insert_value @ 1
addi $r2 $r0 i0               ; store get offset
mcpi $r2 $r1 i16              ; store value
addi $r1 $r0 i0               ; get_ptr
lw   $r2 data_2               ; literal instantiation
sw   $r1 $r2 i0               ; insert_value @ 0
addi $r1 $r0 i0               ; get_ptr
lw   $r0 $r1 i1               ; extract_value @ 1
ret  $r0
noop                          ; word-alignment of data section
.data:
data_0 .u64 0x28
data_1 .u64 0x02
data_2 .u64 0x32
"#;
        simple_test(input, expected);
    }

    #[test]
    fn simple_enum() {
        let input = r#"script script {
    fn main() -> () {
        local ptr { u64, { () | () | u64 } } lunch

        entry:
        v0 = const { u64, { () | () | u64 } } { u64 undef, { () | () | u64 } undef }
        v1 = const u64 1
        v2 = insert_value v0, { u64, { () | () | u64 } }, v1, 0
        store v2, ptr { u64, { () | () | u64 } } lunch
        v3 = get_ptr ptr { u64, { () | () | u64 } } lunch
        v4 = const bool false
        br block0

        block0:
        v5 = phi(entry: v4)
        v6 = const { u64, { () | () | u64 } } { u64 undef, { () | () | u64 } undef }
        v7 = const u64 2
        v8 = insert_value v6, { u64, { () | () | u64 } }, v7, 0
        v9 = const u64 3
        v10 = insert_value v8, { u64, { () | () | u64 } }, v9, 1
        v4 = const bool false
        br block1

        block1:
        v11 = phi(block0: v4)
        ret () v11
    }

}"#;

        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
move $r0 $sp
cfei i16
move $r1 $sp
cfei i16
lw   $r2 data_0               ; literal instantiation
sw   $r1 $r2 i0               ; insert_value @ 0
addi $r2 $r0 i0               ; store get offset
mcpi $r2 $r1 i16              ; store value
addi $r1 $r0 i0               ; get_ptr
lw   $r0 data_1               ; literal instantiation
move $r1 $r0
move $r0 $sp
cfei i16
lw   $r1 data_2               ; literal instantiation
sw   $r0 $r1 i0               ; insert_value @ 0
lw   $r1 data_3               ; literal instantiation
sw   $r0 $r1 i1               ; insert_value @ 1
lw   $r0 data_1               ; literal instantiation
move $r1 $r0
ret  $zero                    ; returning unit as zero
noop                          ; word-alignment of data section
.data:
data_0 .u64 0x01
data_1 .bool 0x00
data_2 .u64 0x02
data_3 .u64 0x03
"#;
        simple_test(input, expected);
    }

    #[test]
    fn lazy_binops() {
        let input = r#"script script {
    fn main() -> bool {
        entry:
        v0 = const bool false
        v0 = const bool false
        cbr v0, block0, block1

        block0:
        v1 = phi(entry: v0)
        v2 = const bool true
        br block1

        block1:
        v3 = phi(entry: v0, block0: v2)
        cbr v3, block3, block2

        block2:
        v4 = phi(block1: v3)
        v5 = const bool true
        br block3

        block3:
        v6 = phi(block1: v3, block2: v5)
        ret bool v6
    }
}"#;

        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
lw   $r0 data_0               ; literal instantiation
move $r1 $r0
move $r1 $r0
jnei $r0 $one i12
lw   $r0 data_1               ; literal instantiation
move $r1 $r0
move $r0 $r1
move $r2 $r1
jnei $r1 $one i16
ji   i18
lw   $r1 data_1               ; literal instantiation
move $r0 $r1
ret  $r0
.data:
data_0 .bool 0x00
data_1 .bool 0x01
"#;
        simple_test(input, expected);
    }

    #[test]
    fn simple_array() {
        let input = r#"script script {
    fn main() -> bool {
        local ptr [bool; 3] a

        entry:
        v0 = const [bool; 3] [bool undef, bool undef, bool undef]
        v1 = const bool false
        v2 = const u64 0
        v3 = insert_element v0, [bool; 3], v1, v2
        v4 = const bool true
        v5 = const u64 1
        v6 = insert_element v3, [bool; 3], v4, v5
        v7 = const bool false
        v8 = const u64 2
        v9 = insert_element v6, [bool; 3], v7, v8
        store v9, ptr [bool; 3] a
        v10 = load ptr [bool; 3] a
        v11 = const u64 1
        v12 = extract_element v10, [bool; 3], v11
        ret bool v12
    }
}"#;

        let expected = r#".program:
ji   i4
noop
DATA_SECTION_OFFSET[0..32]
DATA_SECTION_OFFSET[32..64]
lw   $ds $is 1
add  $$ds $$ds $is
move $r0 $sp
cfei i24
move $r1 $sp
cfei i24
lw   $r2 data_0               ; literal instantiation
lw   $r3 data_1               ; literal instantiation
muli $r3 $r3 i8               ; insert_element relative offset
add  $r4 $r1 $r3              ; insert_element absolute offset
sw   $r4 $r2 i0               ; insert_element
lw   $r2 data_2               ; literal instantiation
lw   $r3 data_3               ; literal instantiation
muli $r3 $r3 i8               ; insert_element relative offset
add  $r4 $r1 $r3              ; insert_element absolute offset
sw   $r4 $r2 i0               ; insert_element
lw   $r2 data_0               ; literal instantiation
lw   $r3 data_4               ; literal instantiation
muli $r3 $r3 i8               ; insert_element relative offset
add  $r4 $r1 $r3              ; insert_element absolute offset
sw   $r4 $r2 i0               ; insert_element
addi $r2 $r0 i0               ; store get offset
mcpi $r2 $r1 i24              ; store value
addi $r1 $r0 i0               ; load address
lw   $r0 data_3               ; literal instantiation
muli $r0 $r0 i8               ; extract_element relative offset
add  $r2 $r1 $r0              ; extract_element absolute offset
lw   $r0 $r2 i0               ; extract_element
ret  $r0
.data:
data_0 .bool 0x00
data_1 .u64 0x00
data_2 .bool 0x01
data_3 .u64 0x01
data_4 .u64 0x02
"#;
        simple_test(input, expected);
    }

    //    #[test]
    //    fn test_name() {
    //        let input = r#"script script {
    //}"#;
    //
    //        let expected = r#".program:
    //"#;
    //        simple_test(input, expected);
    //    }
}

// =================================================================================================
