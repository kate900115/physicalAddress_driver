//++
//
// FACILITY:	FpgaDriverAPI DLL - Communicate with the Catapult device driver
//
// DESCRIPTION:	This module contains public definitions used to interface to the Catapult driver
//
//--

//+
// INCLUDE FILES:
//-

//+
// CONSTANTS:
//-

//+
// MACROS:
//-

//+
// DEFINITIONS:
//-
#define	STATUS_SUCCESS					((NTSTATUS)0x00000000L)
#define STATUS_CONNECTION_IN_USE		((NTSTATUS)0xC0000108L)

inline void FastFence()
{
    __sync_synchronize();
}

//+
// TYPES:
//-

typedef NTSTATUS HW_interrupt_callback (PVOID64 Parameter);
typedef HW_interrupt_callback *pHW_interrupt_callback;


//+
// This structure contains the data within our Buffer descriptor 
//-

typedef struct
	{
	PVOID64				user_address;					// Address of where the memory is mapped in the process
	PVOID64				kernel_address;					// Address of where the memory is mapped in system address space
	PVOID64				physical_address;				// Physical address of the buffer
	ULONG				length;							// Allocated length of the buffer
	} CATAPULT_BUFFER_DESC, *pCATAPULT_BUFFER_DESC;


//+
// DECLARATIONS:
//-

//+
// EXTERNALS:
//-

//+
// GLOBAL ROUTINES:
//-
__forceinline
DWORD APIENTRY HW_register_read32						// Read a 32-bit register
	(
	_In_ _Notliteral_ volatile	DWORD	*Register		// Address of the mapped register
	)

//++
//
// DESCRIPTION:		Read a 32-bit value from a device register. This routine uses the
//					_ReadWriteBarrier compiler intrinsic which prevents the compiler
//					from re-ordering across it and forces reads and writes to memory
//					to complete at the point of the invocation
//
// ASSUMPTIONS:		User mode
//
// RETURN VALUES:
//
//  Contents of the register
//
// SIDE EFFECTS:
//
//--

{
	FastFence ();
	DWORD out = *Register;
	return out;
}									// End routine HW_register_read32


__forceinline
VOID APIENTRY HW_register_write32						// Write a 32-bit register
	(
	_In_ _Notliteral_ volatile	DWORD*	Register,		// Address of the mapped register
	_In_	DWORD						Value			// Value to write
	)

//++
//
// DESCRIPTION:		Write a 32-bit value to a device register. This routine uses the
//					FastFence (__faststorefence) compiler intrinsic which guarantees
//					that every previous memory reference, including both load and store
//					memory references, is globally visible before any subsequent memory
//					reference. On the x64 platform, this generates an instruction that
//					is a faster store fence than the SFENCE instruction
//
// ASSUMPTIONS:		User mode
//
// RETURN VALUES:
//
//		None
//
// SIDE EFFECTS:
//
//--

{
	*Register = Value;
	FastFence ();

	return;
}									// End routine HW_register_write32


__forceinline
DWORD64 APIENTRY HW_register_read64						// Read a 64-bit register
(
_In_ _Notliteral_ volatile	DWORD64	*Register		// Address of the mapped register
)

//++
//
// DESCRIPTION:		Read a 64-bit value from a device register. This routine uses the
//					_ReadWriteBarrier compiler intrinsic which prevents the compiler
//					from re-ordering across it and forces reads and writes to memory
//					to complete at the point of the invocation
//
// ASSUMPTIONS:		User mode
//
// RETURN VALUES:
//
//  Contents of the register
//
// SIDE EFFECTS:
//
//--

{
	FastFence ();
	return *Register;
}									// End routine HW_register_read32


__forceinline
VOID APIENTRY HW_register_write64						// Write a 64-bit register
(
_In_ _Notliteral_ volatile	DWORD64*	Register,		// Address of the mapped register
_In_	DWORD64						Value			// Value to write
)

//++
//
// DESCRIPTION:		Write a 64-bit value to a device register. This routine uses the
//					FastFence (__faststorefence) compiler intrinsic which guarantees
//					that every previous memory reference, including both load and store
//					memory references, is globally visible before any subsequent memory
//					reference. On the x64 platform, this generates an instruction that
//					is a faster store fence than the SFENCE instruction
//
// ASSUMPTIONS:		User mode
//
// RETURN VALUES:
//
//		None
//
// SIDE EFFECTS:
//
//--

{
	*Register = Value;
	FastFence();

	return;
}									// End routine HW_register_write32

/*
__forceinline
VOID APIENTRY HW_register_write128						// Write two 64-bit registers
(
_In_ _Notliteral_ volatile	DWORD64*	Register,		// Address of the mapped register
_In_	DWORD64					*Value			// the pointer of Values to write
)

//++
//
// DESCRIPTION:		Write two 64-bit values to a device register. This routine uses the
//					FastFence (__faststorefence) compiler intrinsic which guarantees
//					that every previous memory reference, including both load and store
//					memory references, is globally visible before any subsequent memory
//					reference. On the x64 platform, this generates an instruction that
//					is a faster store fence than the SFENCE instruction
//
// ASSUMPTIONS:		User mode
//
// RETURN VALUES:
//
//		None
//
// SIDE EFFECTS:
//
//--

{
        __m128i ymm0 = _mm_load_si128((__m128i*)Value);
	_mm_stream_si128((__m128i*)Register, ymm0);

	FastFence();

	return;
}									// End routine HW_register_write128

__forceinline
VOID APIENTRY HW_register_write256						// Write four 64-bit registers
(
_In_ _Notliteral_ volatile	DWORD64*	Register,		// Address of the mapped register
_In_	DWORD64					*Value			// the pointer of Values to write
)

//++
//
// DESCRIPTION:		Write four 64-bit values to a device register. This routine uses the
//					FastFence (__faststorefence) compiler intrinsic which guarantees
//					that every previous memory reference, including both load and store
//					memory references, is globally visible before any subsequent memory
//					reference. On the x64 platform, this generates an instruction that
//					is a faster store fence than the SFENCE instruction
//
// ASSUMPTIONS:		User mode
//
// RETURN VALUES:
//
//		None
//
// SIDE EFFECTS:
//
//--

{
        __m256i ymm0 = _mm256_load_si256((__m256i *) Value);
	_mm256_stream_si256((__m256i *)Register, ymm0);

	FastFence();

	return;
}									// End routine HW_register_write128
*/
