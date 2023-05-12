#include "ut_common.hpp"
#include <macis/bitset_operations.hpp>
#include <iostream>

template <size_t N>
void uint128_conversion_test() {
  using ref_type = macis::uint128_t;
  std::bitset<N> one_bs{1};
  std::bitset<N> val_bs{0};
  val_bs |= (one_bs << 4); 
  if(N > 32) val_bs |= (one_bs << 33);
  if(N > 64) {
    val_bs |= (one_bs << 65);
    val_bs |= (one_bs << 115);
  }

  ref_type one{1};
  ref_type val{0};
  val |= (one << 4); 
  if(N > 32) val |= (one << 33);
  if(N > 64) {
    val |= (one << 65);
    val |= (one << 115);
  }

  auto val_cmp = macis::to_uint128(val_bs);
  REQUIRE( val_cmp == val );
}

template <size_t N> using bs = std::bitset<N>;

TEST_CASE("Bitset Operations") {

  ROOT_ONLY(MPI_COMM_WORLD);

  SECTION("INT128") {
    using macis::uint128_t;
    using ref_type = macis::uint128_t;

    SECTION("TYPE") {
      REQUIRE( std::is_same_v<uint128_t,ref_type> );
    }

    SECTION("From bitset<128>") {
      uint128_conversion_test<128>();      
    }
    SECTION("From bitset<64>") {
      uint128_conversion_test<64>();      
    }
    SECTION("From bitset<32>") {
      uint128_conversion_test<32>();      
    }
    
  }


  SECTION("Full Mask") {

    bs<128> full_128 = ~bs<128>(0);
    bs<128> full_64  = bs<128>(UINT64_MAX);
    bs<128> full_32  = bs<128>(UINT32_MAX);

    SECTION("Static") {
      REQUIRE( full_128 == macis::full_mask<128>()    );
      REQUIRE( full_64  == macis::full_mask<64,128>() );
      REQUIRE( full_32  == macis::full_mask<32,128>() );
    }

    SECTION("Dynamic") {
      REQUIRE( full_128 == macis::full_mask<128>(128) );
      REQUIRE( full_64  == macis::full_mask<128>(64)  );
      REQUIRE( full_32  == macis::full_mask<128>(32)  );
    }

  }

  SECTION("FFS") {

    SECTION("Zero"){ REQUIRE( macis::ffs(bs<64>())  == 0 ); }
    SECTION("One") { REQUIRE( macis::ffs(bs<64>(1)) == 1 ); }
    SECTION("Arbitrary") {
      bs<64> a(0x0A0A); REQUIRE( macis::ffs(a) == 2 );
      bs<32> b(0x0A0A); REQUIRE( macis::ffs(b) == 2 );
      bs<128> c(0xABC); 
        REQUIRE( macis::ffs(c) == 3 );
        REQUIRE( macis::ffs(c << 64) == 3 + 64);
      bs<256> d(0xABC); 
        REQUIRE( macis::ffs(d) == 3 );
        REQUIRE( macis::ffs(d << 64)  == 3 + 64);
        REQUIRE( macis::ffs(d << 128) == 3 + 128);
    }

  }

  SECTION("FLS") {
    bs<64> a(0x0A0A); REQUIRE( macis::fls(a) == 12 - 1 );
    bs<32> b(0x0A0A); REQUIRE( macis::fls(b) == 12 - 1 );
    bs<128> c(0xDEADBEEF); 
      REQUIRE( macis::fls(c) == 31 );
      REQUIRE( macis::fls(c << 64) == 31 + 64 );
    bs<256> d(0xDEADBEEF); 
      REQUIRE( macis::fls(d) == 31 );
      REQUIRE( macis::fls(d << 64)  == 31 + 64 );
      REQUIRE( macis::fls(d << 128) == 31 + 128 );
  }

  SECTION("Indices") {

    bs<128> one(1);
    bs<128> a = (one << 4) | (one << 67) | (one << 118) | (one << 31);
    auto ind = macis::bits_to_indices(a);
    std::vector<uint32_t> ref = {4,31,67,118};
    REQUIRE( ind == ref );

  }


  SECTION("Truncate") {

    bs<64> a_64(0xCCCCCCCCDEADDEAD);
    bs<32> ref (0xDEADDEAD);
    REQUIRE( macis::truncate_bitset<32>(a_64) == ref );

    
  }

  SECTION("Expand") {

    bs<32> a_32(0xDEADDEAD);
    bs<64> ref (0x00000000DEADDEAD);
    REQUIRE( macis::expand_bitset<64>(a_32) == ref );

  }

  SECTION("Compare") {
    SECTION("32 bit") {
      bs<32> a(65), b(42); 
      REQUIRE( macis::bitset_less(b, a) );
      REQUIRE_FALSE( macis::bitset_less(a, b) );
    }
    SECTION("64 bit") {
      bs<64> a(65ull << 32), b(42ull << 32); 
      REQUIRE( macis::bitset_less(b, a) );
      REQUIRE_FALSE( macis::bitset_less(a, b) );
    }
    SECTION("128 bit") {
      bs<128> a(65ull), b(42ull); a = a << 64; b = b << 64;
      REQUIRE( macis::bitset_less(b, a) );
      REQUIRE_FALSE( macis::bitset_less(a, b) );
    }
    SECTION("256 bit") {
      bs<256> a(65ull), b(42ull); a = a << 128; b = b << 128;
      REQUIRE( macis::bitset_less(b, a) );
      REQUIRE_FALSE( macis::bitset_less(a, b) );
    }
  }

}
