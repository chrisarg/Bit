/*
    A simple uncompressed bitset implementation based on David Hanson's
    "C Interfaces and Implementations" book
    Provides an interface to a BIT_T type and a set of functions to
    manipulate it when mapping biological sequences.

    This is not a general bitset library, i.e. one cannot grow the bitset.
    Bitsets are also limited in capacity to int (at the time of the
    writting the same size as uint32_t). If one needs larger bitsets, then
    they should probably be using roaring or compressed bitsets.

    Functions that create, free or load an externally created bitset into a T.
    * Bit_new           : Create a new bitset with a fixed capacity/length
                          The code uses the best alignment possible for the
                          platform (e.g. 64 bytes on x86_64)
    * Bit_free          : Free the bitset, but does not zero the pointer
    * Bit_free_safe     : Free the bitset and zero the pointer
    *
    Functions that obtain the properties of a bitset:
    * Bit_length        : Return the length (or capacity) of the bitset (bytes)
    * Bit_count         : Count the number of bits set in the bitset
    * size_of_Bit_T     : Return the size of an uninitialized Bit_T(in bytes)
    * Bit_size          : Return the size of the buffer needed to store
                          the bitset structure and the buffer of the individual
                          bits (in bytes) for a bitset of a given length.
    * Bit_buffer_size   : Return the size of the buffer needed to store
                          the individual bits of the bitset (in bytes)
    *
    Functions that facilitate the external memory management of the bitset and
    its contents. These functions can be used to interface with other libraries.
    * Bit_load          : Initialize a bitset using an external buffer that is
                          managed
    * size_of_Bit_T     : Return the size of an uninitialized Bit_T(in bytes)


    Functions that manipulate the bitset:
    * Bit_aset          : Set an array of bits in the bitset to one
    * Bit_bset          : Set a bit in the bitset to one
    * Bit_aclear        : Clear an array of bits in the bitset
    * Bit_bclear        : Clear a bit in the bitset
    * Bit_clear         : Clears a range of bits [lo,hi] in the bitset
    * Bit_get           : Get the value of a bit in the bitset
    * Bit_map           : Applies a function to each bit in the bitset. The
    *                     function *may* change the bitset in place. Note that
    *                     as a function is applied from left to right, the
    *                     changes will be seen by subsequent calls
    * Bit_not           : Inverts a range of bits [lo,hi] in the bitset
    * Bit_put           : Set a bit in the bitset to a value & returns the
                          previous value of the bit
    * Bit_set           : Sets a range of bits [lo,hi] in the bitset to one

    Functions that compare bitsets:
    * Bit_eq            : Compare two bitsets for equality (=1)
    * Bit_leq           : Compare two bitsets for less than or equal (=1)
    * Bit_lt            : Compare two bitsets for less than (=1)

    Functions that operate on sets of bitsets (and create a new one):
    * Bit_inter         : Perform an intersection operation with another bitset
    * Bit_diff          : Perform a difference operation, logical AND
    * Bit_minus         : Perform a symmetric difference operation, ie the XOR
    * Bit_union         : Perform a union operation with another bitset


    Functions that perform counts on set operations of two bitsets:
    * Bit_diff_count    : Count the number of bits set in the difference
    * Bit_inter_count   : Count the number of bits set in the intersection
    * Bit_minus_count   : Count the number of bits set in the symmetric
                          difference
    * Bit_union_count   : Count the number of bits set in the union


    * Author : Christos Argyropoulos
    * Created : April 1st 2025
    * License : Free
*/

#ifndef BIT_INCLUDED
#define BIT_INCLUDED

#include <stddef.h>

#define T Bit_T
typedef struct T *T;

/*
    Functions that create, free and obtain the properties of the bitset. Note
    the following error checking
    * Bit_new           : Checked runtime error if length is less than 0 or
                          greater than INT_MAX.
    * Bit_free          : It is a checked runtime error to try to free a bitset
                            that was not allocated by the library.
    * Bit_free_safe     : It is a checked runtime error to try to free a bitset
                          that was not allocated by the library.

    It is a checked runtime error to pass a NULL set to any of these routines.
*/
extern T Bit_new(int length);      // create a new bitset
extern void Bit_free(T set);       // free the bitset
extern void Bit_free_safe(T *set); // free the bitset and zero the pointer

/*
    Functions that obtain the properties of a bitset:
    Note the following error checking:
    Bit_size          : Checked runtime error if length is less than 0 or
                          greater than INT_MAX.
    Bit_buffer_size   : Checked runtime error if length is less than 0 or
                          greater than INT_MAX.

    It is a checked runtime error to pass a NULL set to any of these routines.
*/

extern int Bit_length(T set);
extern int Bit_count(T set);     // uses popcount
extern int size_of_Bit_T(void);  // size of the struct T
extern int Bit_size(int length); // total size needed to store struct T
                                 // and the buffer of bits (in bytes)
extern int Bit_buffer_size(
    int length); // total size needed to store struct T and the buffer

/*
    Functions that manipulate an individual bitset (member operations).
    Note the following error checking:

    It is a checked runtime error to 1) pass a NULL set 2) of the low bit
    is less than zero, 3) the high bit to be greater than the bitset
    length and 4) the low bit to be greater than the high bit, 5) the
    indices to attempt to overrun the bitset length.
    */
extern void Bit_aset(T set, int indices[], int n); // set an array of bits
extern void Bit_bset(T set, int index);   // set a bit in the bitset to 1
extern void Bit_aclear(T set, int indices[],
                      int n); // clear an array of bits in the bitset
extern void Bit_bclear(T set, int index); // clear a bit in the bitset
extern void Bit_clear(T set, int lo,
                      int hi); // clear a range of bits [lo,hi] in the bitset
extern int Bit_get(T set, int index); // returns the bit at index
extern void
Bit_map(T set, void apply(int n, int bit, void *cl),
        void *cl); // maps apply to bit n in the range [0, length-1], where *cl
                   // is a pointer to a closure that is provided by the clinet
extern void Bit_not(T set, int lo,
                    int hi); // inverts a range of bits [lo,hi] in the bitset
extern int Bit_put(T set, int n, int val); // sets the nth bit to val in set
extern void Bit_set(T set, int lo,
                    int hi); // sets a range of bits [lo,hi] in the bitset

/*
    Functions that compare two bitsets; note the following error checking:

    It is a checked runtime error for the two bitsets to be of different
    lengths, or if s or t are NULL.

*/
extern int Bit_eq(T s, T t);  // compare two bitsets for equality
extern int Bit_leq(T s, T t); // compare two bitsets for less than or equal
extern int Bit_lt(T s, T t);  // compare two bitsets for less than

/*
    Functions that operate on sets of bitsets (and create a new one):
    It is a checked runtime error for the two bitsets to be of different
    lengths, or if both s and t are NULL.
    If one of the bitsets is NULL, then the corresponding operation is
    interpreted as against the empty set. In particularBit_

        > Bit_diff(s,NULL) or Bit_diff(NULL, t) returns t or s
        > Bit_diff(s,s) returns (a copy of of) the empty set
        > Bit_inter(s,NULL) or Bit_inter(NULL, t) returns the empty set
        > Bit_minus(NULL,s) or Bit_minus(s,s) returns the empty set
        > Bit_minus(s,NULL) returns a copy of s
        > Bit_union(s,NULL) or Bit_union(NULL, t) makes a copy of s or t
        > Bit_union(s,s) returns a copy of s


*/
extern T Bit_diff(T s, T t);  // difference of two bitsets
extern T Bit_inter(T s, T t); // intersection of two bitsets
extern T Bit_minus(T s, T t); // symmetric difference of two bitsets
extern T Bit_union(T s, T t); // union of two bitsets

/*
    Functions that calculate population counts on the operations of sets of
    bitsets (but without creating a new bitset):
    It is a checked runtime error for the two bitsets to be of different
    lengths, or if both s and t are NULL.
    If one of the bitsets is NULL, then the corresponding operation is
    interpreted as against the empty set. In particularBit_

        > Bit_diff_count(s,NULL) or Bit_diff_count(NULL, t) returns t or s
        > Bit_diff_count(s,s) returns (a copy of of) the empty set
        > Bit_inter_count(s,NULL) or Bit_inter_count(NULL, t) returns the empty
   set 
        > Bit_minus_count(NULL,s) or Bit_minus_count(s,s) returns 0
        > Bit_minus_count(s,NULL) returns Bit_count(s)
        > Bit_union_count(s,NULL) or Bit_union_count(NULL, t)
        > Bit_union_count(s,s) returns a copy of s


*/
extern int Bit_diff_count(T s, T t);  // difference of two bitsets
extern int Bit_inter_count(T s, T t); // intersection of two bitsets
extern int Bit_minus_count(T s, T t); // symmetric difference of two bitsets
extern int Bit_union_count(T s, T t); // union of two bitsets


#undef T
#endif
