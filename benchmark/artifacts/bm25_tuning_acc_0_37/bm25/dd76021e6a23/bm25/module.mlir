module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi32>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu", "polygeist.target-cpu" = "generic", "polygeist.target-features" = "+fp-armv8,+neon,+outline-atomics,+v8a,-fmv"} {
  llvm.mlir.global internal constant @str13("\0AComputation time: %.3f ms\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("An error occurred during ranking.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Rank %d: Doc %d (Score: %.4f) - \22%s\22\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("Ranking results:\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Loaded %d documents\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("Failed to read documents from file.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("Reading documents from: %s\0A\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("Query: \22%s\22\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str5("  query: BM25 query string\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("  documents_file: File containing documents (one per line)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("Usage: %s <documents_file> <query>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2(" .,;:!?\22'\0A\09()[]{}<>\00") {addr_space = 0 : i32}
  memref.global @printed_doc : memref<1xi32> = uninitialized
  memref.global @B : memref<1xf64> = dense<7.500000e-01>
  memref.global @K1 : memref<1xf64> = dense<1.500000e+00>
  llvm.mlir.global internal constant @str1("Error: Cannot open file %s\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str0("r\00") {addr_space = 0 : i32}
  func.func @approx_state_identity(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    return %arg0 : i32
  }
  func.func @compare_scores(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.getelementptr %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
    %4 = llvm.load %3 : !llvm.ptr -> f64
    %5 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
    %6 = llvm.load %5 : !llvm.ptr -> f64
    %7 = arith.cmpf ogt, %4, %6 : f64
    %8 = arith.select %7, %c1_i32, %0 : i32
    %9:2 = scf.if %7 -> (i1, i32) {
      scf.yield %false, %8 : i1, i32
    } else {
      %11 = llvm.getelementptr %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
      %12 = llvm.load %11 : !llvm.ptr -> f64
      %13 = llvm.getelementptr %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
      %14 = llvm.load %13 : !llvm.ptr -> f64
      %15 = arith.cmpf olt, %12, %14 : f64
      %16 = arith.xori %15, %true : i1
      %17 = arith.select %15, %c-1_i32, %8 : i32
      scf.yield %16, %17 : i1, i32
    }
    %10 = arith.select %9#0, %c0_i32, %9#1 : i32
    return %10 : i32
  }
  func.func @compare_tokens(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = call @strcasecmp(%arg0, %arg1) : (memref<?xi8>, memref<?xi8>) -> i32
    return %0 : i32
  }
  func.func private @strcasecmp(memref<?xi8>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @__internal_count_and_lower_words(%arg0: memref<?xi8>, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.icmp "eq" %1, %0 : !llvm.ptr
    %3 = scf.if %2 -> (i1) {
      scf.yield %true : i1
    } else {
      %5 = affine.load %arg0[0] : memref<?xi8>
      %6 = arith.extui %5 : i8 to i32
      %7 = arith.cmpi eq, %6, %c0_i32 : i32
      scf.yield %7 : i1
    }
    %4 = scf.if %3 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %5:4 = scf.while (%arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %arg1, %arg6 = %arg0) : (i32, i32, i32, memref<?xi8>) -> (i32, i32, i32, memref<?xi8>) {
        %6 = arith.cmpi ne, %arg5, %c0_i32 : i32
        scf.condition(%6) %arg4, %arg3, %arg5, %arg6 : i32, i32, i32, memref<?xi8>
      } do {
      ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: memref<?xi8>):
        %6 = affine.load %arg6[0] : memref<?xi8>
        %7 = arith.extui %6 : i8 to i32
        %8 = func.call @tolower(%7) : (i32) -> i32
        %9 = arith.trunci %8 : i32 to i8
        affine.store %9, %arg6[0] : memref<?xi8>
        %10 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
        %11 = affine.load %10[0] : memref<?xmemref<?xi16>>
        %12 = affine.load %arg6[0] : memref<?xi8>
        %13 = arith.extui %12 : i8 to i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = memref.load %11[%14] : memref<?xi16>
        %16 = arith.extui %15 : i16 to i32
        %17 = arith.andi %16, %c8_i32 : i32
        %18 = arith.cmpi ne, %17, %c0_i32 : i32
        %19 = arith.cmpi eq, %arg4, %c0_i32 : i32
        %20 = arith.select %19, %c1_i32, %arg4 : i32
        %21 = arith.andi %18, %19 : i1
        %22 = arith.select %18, %20, %c0_i32 : i32
        %23 = scf.if %21 -> (i32) {
          %26 = arith.addi %arg3, %c1_i32 : i32
          scf.yield %26 : i32
        } else {
          scf.yield %arg3 : i32
        }
        %24 = "polygeist.subindex"(%arg6, %c1) : (memref<?xi8>, index) -> memref<?xi8>
        %25 = arith.addi %arg5, %c-1_i32 : i32
        scf.yield %22, %23, %25, %24 : i32, i32, i32, memref<?xi8>
      }
      scf.yield %5#0 : i32
    }
    return %4 : i32
  }
  func.func @count_and_lower_words(%arg0: memref<?xi8>, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = call @approx_state_identity(%arg2) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c32_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = scf.index_switch %3 -> i32 
    case 0 {
      %5 = func.call @__internal_count_and_lower_words(%arg0, %arg1, %arg2) : (memref<?xi8>, i32, i32) -> i32
      scf.yield %5 : i32
    }
    case 1 {
      %5 = func.call @approx_count_and_lower_words_1(%arg0, %arg1, %arg2) : (memref<?xi8>, i32, i32) -> i32
      scf.yield %5 : i32
    }
    default {
      %5 = func.call @__internal_count_and_lower_words(%arg0, %arg1, %arg2) : (memref<?xi8>, i32, i32) -> i32
      scf.yield %5 : i32
    }
    return %4 : i32
  }
  func.func private @tolower(i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @__ctype_b_loc() -> memref<?xmemref<?xi16>> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @approx_count_and_lower_words_1(%arg0: memref<?xi8>, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.icmp "eq" %1, %0 : !llvm.ptr
    %3 = scf.if %2 -> (i1) {
      scf.yield %true : i1
    } else {
      %5 = affine.load %arg0[0] : memref<?xi8>
      %6 = arith.extui %5 : i8 to i32
      %7 = arith.cmpi eq, %6, %c0_i32 : i32
      scf.yield %7 : i1
    }
    %4 = scf.if %3 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %5 = arith.addi %arg1, %c1_i32 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7:3 = scf.for %arg3 = %c0 to %6 step %c1 iter_args(%arg4 = %c0_i32, %arg5 = %c0_i32, %arg6 = %arg0) -> (i32, i32, memref<?xi8>) {
        %8 = affine.load %arg6[0] : memref<?xi8>
        %9 = arith.extui %8 : i8 to i32
        %10 = arith.cmpi ne, %9, %c32_i32 : i32
        %11 = arith.cmpi eq, %arg4, %c0_i32 : i32
        %12 = arith.select %11, %c1_i32, %arg4 : i32
        %13 = arith.andi %10, %11 : i1
        %14 = arith.select %10, %12, %c0_i32 : i32
        %15 = scf.if %13 -> (i32) {
          %17 = affine.load %arg6[0] : memref<?xi8>
          %18 = arith.extui %17 : i8 to i32
          %19 = func.call @tolower(%18) : (i32) -> i32
          %20 = arith.trunci %19 : i32 to i8
          affine.store %20, %arg6[0] : memref<?xi8>
          %21 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %21 : i32
        } else {
          scf.yield %arg5 : i32
        }
        %16 = "polygeist.subindex"(%arg6, %c1) : (memref<?xi8>, index) -> memref<?xi8>
        scf.yield %14, %15, %16 : i32, i32, memref<?xi8>
      }
      scf.yield %7#1 : i32
    }
    return %4 : i32
  }
  func.func @calculate_idf(%arg0: i32, %arg1: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 5.000000e-01 : f64
    %0 = arith.sitofp %arg1 : i32 to f64
    %1 = arith.sitofp %arg0 : i32 to f64
    %2 = arith.subf %0, %1 : f64
    %3 = arith.addf %2, %cst_1 : f64
    %4 = arith.addf %1, %cst_1 : f64
    %5 = arith.divf %3, %4 : f64
    %6 = arith.addf %5, %cst_0 : f64
    %7 = math.log %6 : f64
    %8 = arith.cmpf ogt, %7, %cst : f64
    %9 = arith.select %8, %7, %cst : f64
    return %9 : f64
  }
  func.func @tf_count_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %7 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
      %8 = llvm.icmp "eq" %7, %2 : !llvm.ptr
      scf.yield %8 : i1
    }
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6 = scf.if %4 -> (i32) {
      scf.yield %5 : i32
    } else {
      %7 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %8 = func.call @strlen(%7) : (!llvm.ptr) -> i64
      %9 = arith.cmpi ne, %8, %c0_i64 : i64
      %10 = scf.if %9 -> (i32) {
        %11 = llvm.mlir.zero : !llvm.ptr
        %12 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
        %13 = arith.index_cast %8 : i64 to index
        %14:2 = scf.while (%arg2 = %c0_i32, %arg3 = %arg1) : (i32, memref<?xi8>) -> (i32, memref<?xi8>) {
          %15 = func.call @strstr(%arg3, %arg0) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
          %16 = "polygeist.memref2pointer"(%15) : (memref<?xi8>) -> !llvm.ptr
          %17 = llvm.icmp "ne" %16, %11 : !llvm.ptr
          scf.condition(%17) %arg2, %15 : i32, memref<?xi8>
        } do {
        ^bb0(%arg2: i32, %arg3: memref<?xi8>):
          %15 = "polygeist.memref2pointer"(%arg3) : (memref<?xi8>) -> !llvm.ptr
          %16 = llvm.icmp "eq" %15, %12 : !llvm.ptr
          %17 = scf.if %16 -> (i1) {
            scf.yield %true : i1
          } else {
            %25 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %26 = affine.load %25[0] : memref<?xmemref<?xi16>>
            %27 = llvm.getelementptr %15[-1] : (!llvm.ptr) -> !llvm.ptr, i8
            %28 = llvm.load %27 : !llvm.ptr -> i8
            %29 = arith.extui %28 : i8 to i32
            %30 = arith.index_cast %29 : i32 to index
            %31 = memref.load %26[%30] : memref<?xi16>
            %32 = arith.extui %31 : i16 to i32
            %33 = arith.andi %32, %c8_i32 : i32
            %34 = arith.cmpi eq, %33, %c0_i32 : i32
            scf.yield %34 : i1
          }
          %18 = memref.load %arg3[%13] : memref<?xi8>
          %19 = arith.extui %18 : i8 to i32
          %20 = arith.cmpi eq, %19, %c0_i32 : i32
          %21 = scf.if %20 -> (i1) {
            scf.yield %true : i1
          } else {
            %25 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %26 = affine.load %25[0] : memref<?xmemref<?xi16>>
            %27 = memref.load %arg3[%13] : memref<?xi8>
            %28 = arith.extui %27 : i8 to i32
            %29 = arith.index_cast %28 : i32 to index
            %30 = memref.load %26[%29] : memref<?xi16>
            %31 = arith.extui %30 : i16 to i32
            %32 = arith.andi %31, %c8_i32 : i32
            %33 = arith.cmpi eq, %32, %c0_i32 : i32
            scf.yield %33 : i1
          }
          %22 = arith.andi %17, %21 : i1
          %23 = scf.if %22 -> (i32) {
            %25 = arith.addi %arg2, %c1_i32 : i32
            scf.yield %25 : i32
          } else {
            scf.yield %arg2 : i32
          }
          %24 = "polygeist.subindex"(%arg3, %13) : (memref<?xi8>, index) -> memref<?xi8>
          scf.yield %23, %24 : i32, memref<?xi8>
        }
        scf.yield %14#0 : i32
      } else {
        scf.yield %5 : i32
      }
      scf.yield %10 : i32
    }
    return %6 : i32
  }
  func.func private @strlen(!llvm.ptr) -> i64
  func.func private @strstr(memref<?xi8>, memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @read_documents_from_file(%arg0: memref<?xi8>, %arg1: memref<?xi32>) -> memref<?xmemref<?xi8>> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c4096_i32 = arith.constant 4096 : i32
    %c0_i8 = arith.constant 0 : i8
    %c10_i32 = arith.constant 10 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<4096xi8>
    %alloca_0 = memref.alloca() : memref<memref<?xmemref<?xi8>>>
    %1 = llvm.mlir.addressof @str0 : !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi8>
    %3 = call @fopen(%arg0, %2) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
    %4 = "polygeist.memref2pointer"(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.icmp "eq" %4, %5 : !llvm.ptr
    scf.if %6 {
      %8 = llvm.mlir.addressof @stderr : !llvm.ptr
      %9 = llvm.load %8 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %10 = "polygeist.memref2pointer"(%9) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %11 = llvm.mlir.addressof @str1 : !llvm.ptr
      %12 = llvm.getelementptr %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
      %13 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %14 = llvm.call @fprintf(%10, %12, %13) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %15 = "polygeist.pointer2memref"(%5) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
      affine.store %15, %alloca_0[] : memref<memref<?xmemref<?xi8>>>
    } else {
      %cast = memref.cast %alloca : memref<4096xi8> to memref<?xi8>
      %8 = llvm.mlir.zero : !llvm.ptr
      %9 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %19 = func.call @fgets(%cast, %c4096_i32, %3) : (memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8>
        %20 = "polygeist.memref2pointer"(%19) : (memref<?xi8>) -> !llvm.ptr
        %21 = llvm.icmp "ne" %20, %8 : !llvm.ptr
        scf.condition(%21) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        %19 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %19 : i32
      }
      func.call @rewind(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> ()
      %10 = "polygeist.typeSize"() <{source = memref<?xi8>}> : () -> index
      %11 = arith.extsi %9 : i32 to i64
      %12 = arith.index_cast %10 : index to i64
      %13 = arith.muli %11, %12 : i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = arith.divui %14, %10 : index
      %alloc = memref.alloc(%15) : memref<?xmemref<?xi8>>
      %16 = "polygeist.memref2pointer"(%alloc) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %17 = llvm.mlir.zero : !llvm.ptr
      %18 = llvm.icmp "eq" %16, %17 : !llvm.ptr
      scf.if %18 {
        %19 = func.call @fclose(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
        %20 = "polygeist.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
        affine.store %20, %alloca_0[] : memref<memref<?xmemref<?xi8>>>
      } else {
        %cast_1 = memref.cast %alloca : memref<4096xi8> to memref<?xi8>
        %19 = llvm.mlir.zero : !llvm.ptr
        %20:3 = scf.while (%arg2 = %0, %arg3 = %c0_i32, %arg4 = %true, %arg5 = %true) : (i32, i32, i1, i1) -> (i32, i1, i32) {
          %21 = func.call @fgets(%cast_1, %c4096_i32, %3) : (memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8>
          %22 = "polygeist.memref2pointer"(%21) : (memref<?xi8>) -> !llvm.ptr
          %23 = llvm.icmp "ne" %22, %19 : !llvm.ptr
          %24 = scf.if %23 -> (i1) {
            %26 = arith.cmpi slt, %arg3, %9 : i32
            scf.yield %26 : i1
          } else {
            scf.yield %false : i1
          }
          %25 = arith.andi %24, %arg5 : i1
          scf.condition(%25) %arg3, %arg4, %arg2 : i32, i1, i32
        } do {
        ^bb0(%arg2: i32, %arg3: i1, %arg4: i32):
          %21 = "polygeist.memref2pointer"(%alloca) : (memref<4096xi8>) -> !llvm.ptr
          %22 = func.call @strlen(%21) : (!llvm.ptr) -> i64
          %23 = arith.cmpi sgt, %22, %c0_i64 : i64
          scf.if %23 {
            %35 = arith.addi %22, %c-1_i64 : i64
            %36 = arith.index_cast %35 : i64 to index
            %37 = memref.load %alloca[%36] : memref<4096xi8>
            %38 = arith.extui %37 : i8 to i32
            %39 = arith.cmpi eq, %38, %c10_i32 : i32
            scf.if %39 {
              memref.store %c0_i8, %alloca[%36] : memref<4096xi8>
            }
          }
          %24 = arith.index_cast %arg2 : i32 to index
          %25 = func.call @strdup(%cast_1) : (memref<?xi8>) -> memref<?xi8>
          memref.store %25, %alloc[%24] : memref<?xmemref<?xi8>>
          %26 = arith.index_cast %arg2 : i32 to index
          %27 = memref.load %alloc[%26] : memref<?xmemref<?xi8>>
          %28 = "polygeist.memref2pointer"(%27) : (memref<?xi8>) -> !llvm.ptr
          %29 = llvm.icmp "eq" %28, %19 : !llvm.ptr
          %30 = arith.xori %29, %true : i1
          %31 = arith.andi %30, %arg3 : i1
          %32 = arith.xori %29, %true : i1
          %33 = arith.select %29, %arg2, %arg4 : i32
          %34 = scf.if %29 -> (i32) {
            %35 = arith.index_cast %arg2 : i32 to index
            scf.for %arg5 = %c0 to %35 step %c1 {
              %38 = memref.load %alloc[%arg5] : memref<?xmemref<?xi8>>
              memref.dealloc %38 : memref<?xi8>
            }
            memref.dealloc %alloc : memref<?xmemref<?xi8>>
            %36 = func.call @fclose(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
            %37 = "polygeist.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
            affine.store %37, %alloca_0[] : memref<memref<?xmemref<?xi8>>>
            scf.yield %arg2 : i32
          } else {
            %35 = arith.addi %arg2, %c1_i32 : i32
            scf.yield %35 : i32
          }
          scf.yield %33, %34, %31, %32 : i32, i32, i1, i1
        }
        scf.if %20#1 {
          %21 = func.call @fclose(%3) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
          affine.store %20#0, %arg1[0] : memref<?xi32>
          affine.store %alloc, %alloca_0[] : memref<memref<?xmemref<?xi8>>>
        }
      }
    }
    %7 = affine.load %alloca_0[] : memref<memref<?xmemref<?xi8>>>
    return %7 : memref<?xmemref<?xi8>>
  }
  func.func private @fopen(memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fgets(memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rewind(memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fclose(memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strdup(memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @df_contains_whole_word(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
      %9 = llvm.icmp "eq" %8, %2 : !llvm.ptr
      scf.yield %9 : i1
    }
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6:2 = scf.if %4 -> (i1, i32) {
      scf.yield %false, %5 : i1, i32
    } else {
      %8 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %9 = func.call @strlen(%8) : (!llvm.ptr) -> i64
      %10 = arith.cmpi ne, %9, %c0_i64 : i64
      %11:2 = scf.if %10 -> (i1, i32) {
        %12 = arith.cmpi eq, %9, %c0_i64 : i64
        %13 = arith.select %12, %c0_i32, %5 : i32
        %14 = llvm.mlir.zero : !llvm.ptr
        %15:3 = scf.while (%arg2 = %true, %arg3 = %13, %arg4 = %true, %arg5 = %arg1) : (i1, i32, i1, memref<?xi8>) -> (i1, i32, memref<?xi8>) {
          %16 = func.call @strstr(%arg5, %arg0) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
          %17 = "polygeist.memref2pointer"(%16) : (memref<?xi8>) -> !llvm.ptr
          %18 = llvm.icmp "ne" %17, %14 : !llvm.ptr
          %19 = arith.andi %18, %arg4 : i1
          scf.condition(%19) %arg2, %arg3, %16 : i1, i32, memref<?xi8>
        } do {
        ^bb0(%arg2: i1, %arg3: i32, %arg4: memref<?xi8>):
          %16 = "polygeist.memref2pointer"(%arg4) : (memref<?xi8>) -> !llvm.ptr
          %17 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
          %18 = llvm.icmp "eq" %16, %17 : !llvm.ptr
          %19 = scf.if %18 -> (i1) {
            scf.yield %true : i1
          } else {
            %33 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %34 = affine.load %33[0] : memref<?xmemref<?xi16>>
            %35 = llvm.getelementptr %16[-1] : (!llvm.ptr) -> !llvm.ptr, i8
            %36 = llvm.load %35 : !llvm.ptr -> i8
            %37 = arith.extui %36 : i8 to i32
            %38 = arith.index_cast %37 : i32 to index
            %39 = memref.load %34[%38] : memref<?xi16>
            %40 = arith.extui %39 : i16 to i32
            %41 = arith.andi %40, %c8_i32 : i32
            %42 = arith.cmpi eq, %41, %c0_i32 : i32
            scf.yield %42 : i1
          }
          %20 = arith.extsi %19 : i1 to i32
          %21 = arith.cmpi eq, %20, %c0_i32 : i32
          %22 = arith.index_cast %9 : i64 to index
          %23 = memref.load %arg4[%22] : memref<?xi8>
          %24 = arith.extui %23 : i8 to i32
          %25 = arith.cmpi eq, %24, %c0_i32 : i32
          %26 = scf.if %25 -> (i1) {
            scf.yield %true : i1
          } else {
            %33 = func.call @__ctype_b_loc() : () -> memref<?xmemref<?xi16>>
            %34 = affine.load %33[0] : memref<?xmemref<?xi16>>
            %35 = memref.load %arg4[%22] : memref<?xi8>
            %36 = arith.extui %35 : i8 to i32
            %37 = arith.index_cast %36 : i32 to index
            %38 = memref.load %34[%37] : memref<?xi16>
            %39 = arith.extui %38 : i16 to i32
            %40 = arith.andi %39, %c8_i32 : i32
            %41 = arith.cmpi eq, %40, %c0_i32 : i32
            scf.yield %41 : i1
          }
          %27 = arith.extsi %26 : i1 to i32
          %28 = arith.cmpi eq, %27, %c0_i32 : i32
          %29 = arith.andi %19, %28 : i1
          %30 = arith.ori %29, %21 : i1
          %31:2 = scf.if %19 -> (i1, i32) {
            %33 = arith.cmpi eq, %27, %c0_i32 : i32
            %34 = arith.andi %33, %arg2 : i1
            %35 = arith.select %26, %c1_i32, %arg3 : i32
            scf.yield %34, %35 : i1, i32
          } else {
            scf.yield %arg2, %arg3 : i1, i32
          }
          %32 = scf.if %30 -> (memref<?xi8>) {
            %33 = arith.index_cast %9 : i64 to index
            %34 = "polygeist.subindex"(%arg4, %33) : (memref<?xi8>, index) -> memref<?xi8>
            scf.yield %34 : memref<?xi8>
          } else {
            scf.yield %arg4 : memref<?xi8>
          }
          scf.yield %31#0, %31#1, %30, %32 : i1, i32, i1, memref<?xi8>
        }
        scf.yield %15#0, %15#1 : i1, i32
      } else {
        scf.yield %false, %5 : i1, i32
      }
      scf.yield %11#0, %11#1 : i1, i32
    }
    %7 = arith.select %6#0, %c0_i32, %6#1 : i32
    return %7 : i32
  }
  func.func @calculate_df(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.icmp "eq" %2, %1 : !llvm.ptr
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %7 = "polygeist.memref2pointer"(%arg1) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %8 = llvm.icmp "eq" %7, %1 : !llvm.ptr
      scf.yield %8 : i1
    }
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6 = scf.if %4 -> (i32) {
      scf.yield %5 : i32
    } else {
      %7 = func.call @lower_dup(%arg0) : (memref<?xi8>) -> memref<?xi8>
      %8 = "polygeist.memref2pointer"(%7) : (memref<?xi8>) -> !llvm.ptr
      %9 = llvm.mlir.zero : !llvm.ptr
      %10 = llvm.icmp "eq" %8, %9 : !llvm.ptr
      %11 = scf.if %10 -> (i32) {
        scf.yield %5 : i32
      } else {
        %12 = arith.index_cast %arg2 : i32 to index
        %13 = scf.for %arg3 = %c0 to %12 step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
          %14 = memref.load %arg1[%arg3] : memref<?xmemref<?xi8>>
          %15 = func.call @df_contains_whole_word(%7, %14) : (memref<?xi8>, memref<?xi8>) -> i32
          %16 = arith.cmpi ne, %15, %c0_i32 : i32
          %17 = scf.if %16 -> (i32) {
            %18 = arith.addi %arg4, %c1_i32 : i32
            scf.yield %18 : i32
          } else {
            scf.yield %arg4 : i32
          }
          scf.yield %17 : i32
        }
        memref.dealloc %7 : memref<?xi8>
        scf.yield %13 : i32
      }
      scf.yield %11 : i32
    }
    return %6 : i32
  }
  func.func private @lower_dup(%arg0: memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.icmp "eq" %0, %1 : !llvm.ptr
    %3 = scf.if %2 -> (memref<?xi8>) {
      %4 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi8>
      scf.yield %4 : memref<?xi8>
    } else {
      %4 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
      %5 = func.call @strlen(%4) : (!llvm.ptr) -> i64
      %6 = arith.addi %5, %c1_i64 : i64
      %7 = arith.index_cast %6 : i64 to index
      %alloc = memref.alloc(%7) : memref<?xi8>
      %8 = "polygeist.memref2pointer"(%alloc) : (memref<?xi8>) -> !llvm.ptr
      %9 = llvm.mlir.zero : !llvm.ptr
      %10 = llvm.icmp "eq" %8, %9 : !llvm.ptr
      %11 = scf.if %10 -> (memref<?xi8>) {
        %12 = "polygeist.pointer2memref"(%9) : (!llvm.ptr) -> memref<?xi8>
        scf.yield %12 : memref<?xi8>
      } else {
        %12 = arith.index_cast %5 : i64 to index
        scf.for %arg1 = %c0 to %12 step %c1 {
          %14 = memref.load %arg0[%arg1] : memref<?xi8>
          %15 = arith.extui %14 : i8 to i32
          %16 = func.call @tolower(%15) : (i32) -> i32
          %17 = arith.trunci %16 : i32 to i8
          memref.store %17, %alloc[%arg1] : memref<?xi8>
        }
        %13 = arith.index_cast %5 : i64 to index
        memref.store %c0_i8, %alloc[%13] : memref<?xi8>
        scf.yield %alloc : memref<?xi8>
      }
      scf.yield %11 : memref<?xi8>
    }
    return %3 : memref<?xi8>
  }
  func.func @__internal_score_term_over_docs(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xf64>, %arg3: f64, %arg4: f64, %arg5: memref<?x!llvm.struct<(i32, f64)>>, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %0 = arith.index_cast %arg6 : i32 to index
    scf.for %arg8 = %c0 to %0 step %c1 {
      %1 = memref.load %arg1[%arg8] : memref<?xmemref<?xi8>>
      %2 = func.call @tf_count_whole_word(%arg0, %1) : (memref<?xi8>, memref<?xi8>) -> i32
      %3 = arith.sitofp %2 : i32 to f64
      %4 = memref.get_global @K1 : memref<1xf64>
      %5 = affine.load %4[0] : memref<1xf64>
      %6 = arith.addf %5, %cst_0 : f64
      %7 = arith.mulf %3, %6 : f64
      %8 = memref.get_global @B : memref<1xf64>
      %9 = affine.load %8[0] : memref<1xf64>
      %10 = arith.subf %cst_0, %9 : f64
      %11 = memref.load %arg2[%arg8] : memref<?xf64>
      %12 = arith.divf %11, %arg3 : f64
      %13 = arith.mulf %9, %12 : f64
      %14 = arith.addf %10, %13 : f64
      %15 = arith.mulf %5, %14 : f64
      %16 = arith.addf %3, %15 : f64
      %17 = arith.cmpf ogt, %16, %cst : f64
      %18 = scf.if %17 -> (f64) {
        %27 = arith.divf %7, %16 : f64
        scf.yield %27 : f64
      } else {
        scf.yield %cst : f64
      }
      %19 = arith.mulf %arg4, %18 : f64
      %20 = arith.muli %arg8, %c16 : index
      %21 = arith.index_cast %20 : index to i64
      %22 = "polygeist.memref2pointer"(%arg5) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
      %23 = llvm.getelementptr %22[%21] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %24 = llvm.getelementptr %23[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
      %25 = llvm.load %24 : !llvm.ptr -> f64
      %26 = arith.addf %25, %19 : f64
      llvm.store %26, %24 : f64, !llvm.ptr
    }
    return
  }
  func.func @score_term_over_docs(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xf64>, %arg3: f64, %arg4: f64, %arg5: memref<?x!llvm.struct<(i32, f64)>>, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c33_i32 = arith.constant 33 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = call @approx_state_identity(%arg7) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c33_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.index_switch %3 
    case 0 {
      func.call @approx_score_term_over_docs_1(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (memref<?xi8>, memref<?xmemref<?xi8>>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i32, f64)>>, i32, i32) -> ()
      scf.yield
    }
    case 1 {
      func.call @__internal_score_term_over_docs(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (memref<?xi8>, memref<?xmemref<?xi8>>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i32, f64)>>, i32, i32) -> ()
      scf.yield
    }
    default {
      func.call @approx_score_term_over_docs_1(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (memref<?xi8>, memref<?xmemref<?xi8>>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i32, f64)>>, i32, i32) -> ()
    }
    return
  }
  func.func @approx_score_term_over_docs_1(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xf64>, %arg3: f64, %arg4: f64, %arg5: memref<?x!llvm.struct<(i32, f64)>>, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c10_i32 = arith.constant 10 : i32
    %c100_i32 = arith.constant 100 : i32
    %0 = arith.index_cast %arg6 : i32 to index
    scf.for %arg8 = %c0 to %0 step %c1 {
      %1 = func.call @rand() : () -> i32
      %2 = arith.remsi %1, %c100_i32 : i32
      %3 = arith.cmpi sge, %2, %c10_i32 : i32
      scf.if %3 {
        %4 = memref.load %arg1[%arg8] : memref<?xmemref<?xi8>>
        %5 = func.call @tf_count_whole_word(%arg0, %4) : (memref<?xi8>, memref<?xi8>) -> i32
        %6 = arith.sitofp %5 : i32 to f64
        %7 = memref.get_global @K1 : memref<1xf64>
        %8 = affine.load %7[0] : memref<1xf64>
        %9 = arith.addf %8, %cst_0 : f64
        %10 = arith.mulf %6, %9 : f64
        %11 = arith.sitofp %5 : i32 to f64
        %12 = memref.get_global @K1 : memref<1xf64>
        %13 = affine.load %12[0] : memref<1xf64>
        %14 = memref.get_global @B : memref<1xf64>
        %15 = affine.load %14[0] : memref<1xf64>
        %16 = arith.subf %cst_0, %15 : f64
        %17 = memref.load %arg2[%arg8] : memref<?xf64>
        %18 = arith.divf %17, %arg3 : f64
        %19 = arith.mulf %15, %18 : f64
        %20 = arith.addf %16, %19 : f64
        %21 = arith.mulf %13, %20 : f64
        %22 = arith.addf %11, %21 : f64
        %23 = arith.cmpf ogt, %22, %cst : f64
        %24 = scf.if %23 -> (f64) {
          %33 = arith.divf %10, %22 : f64
          scf.yield %33 : f64
        } else {
          scf.yield %cst : f64
        }
        %25 = arith.mulf %arg4, %24 : f64
        %26 = arith.muli %arg8, %c16 : index
        %27 = arith.index_cast %26 : index to i64
        %28 = "polygeist.memref2pointer"(%arg5) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %29 = llvm.getelementptr %28[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %30 = llvm.getelementptr %29[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        %31 = llvm.load %30 : !llvm.ptr -> f64
        %32 = arith.addf %31, %25 : f64
        llvm.store %32, %30 : f64, !llvm.ptr
      }
    }
    return
  }
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @approx_score_term_over_docs_2(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xf64>, %arg3: f64, %arg4: f64, %arg5: memref<?x!llvm.struct<(i32, f64)>>, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c20_i32 = arith.constant 20 : i32
    %c100_i32 = arith.constant 100 : i32
    %0 = arith.index_cast %arg6 : i32 to index
    scf.for %arg8 = %c0 to %0 step %c1 {
      %1 = func.call @rand() : () -> i32
      %2 = arith.remsi %1, %c100_i32 : i32
      %3 = arith.cmpi sge, %2, %c20_i32 : i32
      scf.if %3 {
        %4 = memref.load %arg1[%arg8] : memref<?xmemref<?xi8>>
        %5 = func.call @tf_count_whole_word(%arg0, %4) : (memref<?xi8>, memref<?xi8>) -> i32
        %6 = arith.sitofp %5 : i32 to f64
        %7 = memref.get_global @K1 : memref<1xf64>
        %8 = affine.load %7[0] : memref<1xf64>
        %9 = arith.addf %8, %cst_0 : f64
        %10 = arith.mulf %6, %9 : f64
        %11 = arith.sitofp %5 : i32 to f64
        %12 = memref.get_global @K1 : memref<1xf64>
        %13 = affine.load %12[0] : memref<1xf64>
        %14 = memref.get_global @B : memref<1xf64>
        %15 = affine.load %14[0] : memref<1xf64>
        %16 = arith.subf %cst_0, %15 : f64
        %17 = memref.load %arg2[%arg8] : memref<?xf64>
        %18 = arith.divf %17, %arg3 : f64
        %19 = arith.mulf %15, %18 : f64
        %20 = arith.addf %16, %19 : f64
        %21 = arith.mulf %13, %20 : f64
        %22 = arith.addf %11, %21 : f64
        %23 = arith.cmpf ogt, %22, %cst : f64
        %24 = scf.if %23 -> (f64) {
          %33 = arith.divf %10, %22 : f64
          scf.yield %33 : f64
        } else {
          scf.yield %cst : f64
        }
        %25 = arith.mulf %arg4, %24 : f64
        %26 = arith.muli %arg8, %c16 : index
        %27 = arith.index_cast %26 : index to i64
        %28 = "polygeist.memref2pointer"(%arg5) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %29 = llvm.getelementptr %28[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %30 = llvm.getelementptr %29[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        %31 = llvm.load %30 : !llvm.ptr -> f64
        %32 = arith.addf %31, %25 : f64
        llvm.store %32, %30 : f64, !llvm.ptr
      }
    }
    return
  }
  func.func @__internal_lowering_corpus(%arg0: memref<?xmemref<?xi8>>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?x!llvm.struct<(i32, f64)>>, %arg5: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = llvm.mlir.undef : f64
    %1 = affine.load %arg2[0] : memref<?xi32>
    %2 = arith.index_cast %1 : i32 to index
    %3 = scf.for %arg6 = %c0 to %2 step %c1 iter_args(%arg7 = %0) -> (f64) {
      %5 = arith.index_cast %arg6 : index to i32
      %6 = memref.load %arg0[%arg6] : memref<?xmemref<?xi8>>
      %7 = func.call @strdup(%6) : (memref<?xi8>) -> memref<?xi8>
      %8 = "polygeist.memref2pointer"(%7) : (memref<?xi8>) -> !llvm.ptr
      %9 = func.call @strlen(%8) : (!llvm.ptr) -> i64
      %10 = arith.trunci %9 : i64 to i32
      %11 = func.call @count_and_lower_words(%7, %10, %10) : (memref<?xi8>, i32, i32) -> i32
      %12 = arith.sitofp %11 : i32 to f64
      memref.store %12, %arg3[%arg6] : memref<?xf64>
      %13 = memref.load %arg3[%arg6] : memref<?xf64>
      %14 = arith.addf %arg7, %13 : f64
      memref.store %7, %arg1[%arg6] : memref<?xmemref<?xi8>>
      %15 = arith.muli %arg6, %c16 : index
      %16 = arith.index_cast %15 : index to i64
      %17 = "polygeist.memref2pointer"(%arg4) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
      %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      llvm.store %5, %18 : i32, !llvm.ptr
      %19 = llvm.getelementptr %18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
      llvm.store %cst, %19 : f64, !llvm.ptr
      scf.yield %14 : f64
    }
    %4 = arith.fptosi %3 : f64 to i32
    return %4 : i32
  }
  func.func @lowering_corpus(%arg0: memref<?xmemref<?xi8>>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?x!llvm.struct<(i32, f64)>>, %arg5: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = call @approx_state_identity(%arg5) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c5_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = scf.index_switch %3 -> i32 
    case 0 {
      %5 = func.call @__internal_lowering_corpus(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (memref<?xmemref<?xi8>>, memref<?xmemref<?xi8>>, memref<?xi32>, memref<?xf64>, memref<?x!llvm.struct<(i32, f64)>>, i32) -> i32
      scf.yield %5 : i32
    }
    case 1 {
      %5 = func.call @__internal_lowering_corpus(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (memref<?xmemref<?xi8>>, memref<?xmemref<?xi8>>, memref<?xi32>, memref<?xf64>, memref<?x!llvm.struct<(i32, f64)>>, i32) -> i32
      scf.yield %5 : i32
    }
    default {
      %5 = func.call @__internal_lowering_corpus(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (memref<?xmemref<?xi8>>, memref<?xmemref<?xi8>>, memref<?xi32>, memref<?xf64>, memref<?x!llvm.struct<(i32, f64)>>, i32) -> i32
      scf.yield %5 : i32
    }
    return %4 : i32
  }
  func.func @approx_lowering_corpus_1(%arg0: memref<?xmemref<?xi8>>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?x!llvm.struct<(i32, f64)>>, %arg5: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c15_i32 = arith.constant 15 : i32
    %c100_i32 = arith.constant 100 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %2:4 = scf.while (%arg6 = %0, %arg7 = %c0_i32, %arg8 = %1, %arg9 = %c0_i32) : (i32, i32, f64, i32) -> (f64, i32, i32, i32) {
      %4 = affine.load %arg2[0] : memref<?xi32>
      %5 = arith.cmpi slt, %arg7, %4 : i32
      scf.condition(%5) %arg8, %arg9, %arg6, %arg7 : f64, i32, i32, i32
    } do {
    ^bb0(%arg6: f64, %arg7: i32, %arg8: i32, %arg9: i32):
      %4 = func.call @rand() : () -> i32
      %5 = arith.remsi %4, %c100_i32 : i32
      %6 = arith.cmpi sge, %5, %c15_i32 : i32
      %7:3 = scf.if %6 -> (i32, f64, i32) {
        %9 = arith.index_cast %arg9 : i32 to index
        %10 = memref.load %arg0[%9] : memref<?xmemref<?xi8>>
        %11 = func.call @strdup(%10) : (memref<?xi8>) -> memref<?xi8>
        %12 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
        %13 = func.call @strlen(%12) : (!llvm.ptr) -> i64
        %14 = arith.trunci %13 : i64 to i32
        %15 = arith.index_cast %arg7 : i32 to index
        %16 = func.call @count_and_lower_words(%11, %14, %14) : (memref<?xi8>, i32, i32) -> i32
        %17 = arith.sitofp %16 : i32 to f64
        memref.store %17, %arg3[%15] : memref<?xf64>
        %18 = arith.index_cast %arg7 : i32 to index
        %19 = memref.load %arg3[%18] : memref<?xf64>
        %20 = arith.addf %arg6, %19 : f64
        %21 = arith.index_cast %arg7 : i32 to index
        memref.store %11, %arg1[%21] : memref<?xmemref<?xi8>>
        %22 = arith.index_cast %arg7 : i32 to index
        %23 = arith.muli %22, %c16 : index
        %24 = arith.index_cast %23 : index to i64
        %25 = "polygeist.memref2pointer"(%arg4) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %26 = llvm.getelementptr %25[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        llvm.store %arg9, %26 : i32, !llvm.ptr
        %27 = arith.index_cast %arg9 : i32 to index
        %28 = arith.muli %27, %c16 : index
        %29 = arith.index_cast %28 : index to i64
        %30 = "polygeist.memref2pointer"(%arg4) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %31 = llvm.getelementptr %30[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %32 = llvm.getelementptr %31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        llvm.store %cst, %32 : f64, !llvm.ptr
        %33 = arith.addi %arg7, %c1_i32 : i32
        scf.yield %14, %20, %33 : i32, f64, i32
      } else {
        scf.yield %arg8, %arg6, %arg7 : i32, f64, i32
      }
      %8 = arith.addi %arg9, %c1_i32 : i32
      scf.yield %7#0, %8, %7#1, %7#2 : i32, i32, f64, i32
    }
    affine.store %2#1, %arg2[0] : memref<?xi32>
    %3 = arith.fptosi %2#0 : f64 to i32
    return %3 : i32
  }
  func.func @approx_lowering_corpus_2(%arg0: memref<?xmemref<?xi8>>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?x!llvm.struct<(i32, f64)>>, %arg5: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c30_i32 = arith.constant 30 : i32
    %c100_i32 = arith.constant 100 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %2:4 = scf.while (%arg6 = %0, %arg7 = %c0_i32, %arg8 = %1, %arg9 = %c0_i32) : (i32, i32, f64, i32) -> (f64, i32, i32, i32) {
      %4 = affine.load %arg2[0] : memref<?xi32>
      %5 = arith.cmpi slt, %arg7, %4 : i32
      scf.condition(%5) %arg8, %arg9, %arg6, %arg7 : f64, i32, i32, i32
    } do {
    ^bb0(%arg6: f64, %arg7: i32, %arg8: i32, %arg9: i32):
      %4 = func.call @rand() : () -> i32
      %5 = arith.remsi %4, %c100_i32 : i32
      %6 = arith.cmpi sge, %5, %c30_i32 : i32
      %7:3 = scf.if %6 -> (i32, f64, i32) {
        %9 = arith.index_cast %arg9 : i32 to index
        %10 = memref.load %arg0[%9] : memref<?xmemref<?xi8>>
        %11 = func.call @strdup(%10) : (memref<?xi8>) -> memref<?xi8>
        %12 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
        %13 = func.call @strlen(%12) : (!llvm.ptr) -> i64
        %14 = arith.trunci %13 : i64 to i32
        %15 = arith.index_cast %arg7 : i32 to index
        %16 = func.call @count_and_lower_words(%11, %14, %14) : (memref<?xi8>, i32, i32) -> i32
        %17 = arith.sitofp %16 : i32 to f64
        memref.store %17, %arg3[%15] : memref<?xf64>
        %18 = arith.index_cast %arg7 : i32 to index
        %19 = memref.load %arg3[%18] : memref<?xf64>
        %20 = arith.addf %arg6, %19 : f64
        %21 = arith.index_cast %arg7 : i32 to index
        memref.store %11, %arg1[%21] : memref<?xmemref<?xi8>>
        %22 = arith.index_cast %arg7 : i32 to index
        %23 = arith.muli %22, %c16 : index
        %24 = arith.index_cast %23 : index to i64
        %25 = "polygeist.memref2pointer"(%arg4) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %26 = llvm.getelementptr %25[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        llvm.store %arg9, %26 : i32, !llvm.ptr
        %27 = arith.index_cast %arg9 : i32 to index
        %28 = arith.muli %27, %c16 : index
        %29 = arith.index_cast %28 : index to i64
        %30 = "polygeist.memref2pointer"(%arg4) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %31 = llvm.getelementptr %30[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %32 = llvm.getelementptr %31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        llvm.store %cst, %32 : f64, !llvm.ptr
        %33 = arith.addi %arg7, %c1_i32 : i32
        scf.yield %14, %20, %33 : i32, f64, i32
      } else {
        scf.yield %arg8, %arg6, %arg7 : i32, f64, i32
      }
      %8 = arith.addi %arg9, %c1_i32 : i32
      scf.yield %7#0, %8, %7#1, %7#2 : i32, i32, f64, i32
    }
    affine.store %2#1, %arg2[0] : memref<?xi32>
    %3 = arith.fptosi %2#0 : f64 to i32
    return %3 : i32
  }
  func.func @approx_lowering_corpus_3(%arg0: memref<?xmemref<?xi8>>, %arg1: memref<?xmemref<?xi8>>, %arg2: memref<?xi32>, %arg3: memref<?xf64>, %arg4: memref<?x!llvm.struct<(i32, f64)>>, %arg5: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c50_i32 = arith.constant 50 : i32
    %c100_i32 = arith.constant 100 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %2:4 = scf.while (%arg6 = %0, %arg7 = %c0_i32, %arg8 = %1, %arg9 = %c0_i32) : (i32, i32, f64, i32) -> (f64, i32, i32, i32) {
      %4 = affine.load %arg2[0] : memref<?xi32>
      %5 = arith.cmpi slt, %arg7, %4 : i32
      scf.condition(%5) %arg8, %arg9, %arg6, %arg7 : f64, i32, i32, i32
    } do {
    ^bb0(%arg6: f64, %arg7: i32, %arg8: i32, %arg9: i32):
      %4 = func.call @rand() : () -> i32
      %5 = arith.remsi %4, %c100_i32 : i32
      %6 = arith.cmpi sge, %5, %c50_i32 : i32
      %7:3 = scf.if %6 -> (i32, f64, i32) {
        %9 = arith.index_cast %arg9 : i32 to index
        %10 = memref.load %arg0[%9] : memref<?xmemref<?xi8>>
        %11 = func.call @strdup(%10) : (memref<?xi8>) -> memref<?xi8>
        %12 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
        %13 = func.call @strlen(%12) : (!llvm.ptr) -> i64
        %14 = arith.trunci %13 : i64 to i32
        %15 = arith.index_cast %arg7 : i32 to index
        %16 = func.call @count_and_lower_words(%11, %14, %14) : (memref<?xi8>, i32, i32) -> i32
        %17 = arith.sitofp %16 : i32 to f64
        memref.store %17, %arg3[%15] : memref<?xf64>
        %18 = arith.index_cast %arg7 : i32 to index
        %19 = memref.load %arg3[%18] : memref<?xf64>
        %20 = arith.addf %arg6, %19 : f64
        %21 = arith.index_cast %arg7 : i32 to index
        memref.store %11, %arg1[%21] : memref<?xmemref<?xi8>>
        %22 = arith.index_cast %arg7 : i32 to index
        %23 = arith.muli %22, %c16 : index
        %24 = arith.index_cast %23 : index to i64
        %25 = "polygeist.memref2pointer"(%arg4) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %26 = llvm.getelementptr %25[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        llvm.store %arg9, %26 : i32, !llvm.ptr
        %27 = arith.index_cast %arg9 : i32 to index
        %28 = arith.muli %27, %c16 : index
        %29 = arith.index_cast %28 : index to i64
        %30 = "polygeist.memref2pointer"(%arg4) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %31 = llvm.getelementptr %30[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %32 = llvm.getelementptr %31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
        llvm.store %cst, %32 : f64, !llvm.ptr
        %33 = arith.addi %arg7, %c1_i32 : i32
        scf.yield %14, %20, %33 : i32, f64, i32
      } else {
        scf.yield %arg8, %arg6, %arg7 : i32, f64, i32
      }
      %8 = arith.addi %arg9, %c1_i32 : i32
      scf.yield %7#0, %8, %7#1, %7#2 : i32, i32, f64, i32
    }
    affine.store %2#1, %arg2[0] : memref<?xi32>
    %3 = arith.fptosi %2#0 : f64 to i32
    return %3 : i32
  }
  func.func @rank_documents_bm25(%arg0: memref<?xi8>, %arg1: memref<?xmemref<?xi8>>, %arg2: i32, %arg3: i32) -> memref<?x!llvm.struct<(i32, f64)>> attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c16 = arith.constant 16 : index
    %c16_i64 = arith.constant 16 : i64
    %cst = arith.constant 1.000000e+01 : f64
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.undef : f64
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %alloca_0 = memref.alloca() : memref<256xmemref<?xi8>>
    %alloca_1 = memref.alloca() : memref<1xi32>
    affine.store %0, %alloca_1[0] : memref<1xi32>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %4 = llvm.icmp "eq" %3, %2 : !llvm.ptr
    %5 = scf.if %4 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = "polygeist.memref2pointer"(%arg1) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %9 = llvm.icmp "eq" %8, %2 : !llvm.ptr
      scf.yield %9 : i1
    }
    %6 = scf.if %5 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = arith.cmpi sle, %arg2, %c0_i32 : i32
      scf.yield %8 : i1
    }
    %7 = scf.if %6 -> (memref<?x!llvm.struct<(i32, f64)>>) {
      %8 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
      scf.yield %8 : memref<?x!llvm.struct<(i32, f64)>>
    } else {
      affine.store %arg2, %alloca_1[0] : memref<1xi32>
      %8 = arith.extsi %arg2 : i32 to i64
      %9 = arith.muli %8, %c8_i64 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = arith.divui %10, %c8 : index
      %alloc = memref.alloc(%11) : memref<?xf64>
      %12 = "polygeist.typeSize"() <{source = memref<?xi8>}> : () -> index
      %13 = arith.extsi %arg2 : i32 to i64
      %14 = arith.index_cast %12 : index to i64
      %15 = arith.muli %13, %14 : i64
      %16 = arith.index_cast %15 : i64 to index
      %17 = arith.divui %16, %12 : index
      %alloc_2 = memref.alloc(%17) : memref<?xmemref<?xi8>>
      %18 = arith.extsi %arg2 : i32 to i64
      %19 = arith.muli %18, %c16_i64 : i64
      %20 = arith.index_cast %19 : i64 to index
      %21 = arith.divui %20, %c16 : index
      %alloc_3 = memref.alloc(%21) : memref<?x!llvm.struct<(i32, f64)>>
      %cast = memref.cast %alloca_1 : memref<1xi32> to memref<?xi32>
      %22 = func.call @lowering_corpus(%arg1, %alloc_2, %cast, %alloc, %alloc_3, %arg3) : (memref<?xmemref<?xi8>>, memref<?xmemref<?xi8>>, memref<?xi32>, memref<?xf64>, memref<?x!llvm.struct<(i32, f64)>>, i32) -> i32
      %23 = arith.sitofp %22 : i32 to f64
      %24 = affine.load %alloca_1[0] : memref<1xi32>
      %25 = arith.sitofp %24 : i32 to f64
      %26 = arith.divf %23, %25 : f64
      %27 = memref.get_global @printed_doc : memref<1xi32>
      affine.store %24, %27[0] : memref<1xi32>
      %28 = func.call @strdup(%arg0) : (memref<?xi8>) -> memref<?xi8>
      %29 = "polygeist.memref2pointer"(%28) : (memref<?xi8>) -> !llvm.ptr
      %30 = llvm.mlir.zero : !llvm.ptr
      %31 = llvm.icmp "eq" %29, %30 : !llvm.ptr
      %32 = scf.if %31 -> (memref<?x!llvm.struct<(i32, f64)>>) {
        %33 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
          %35 = affine.load %alloca_1[0] : memref<1xi32>
          %36 = arith.cmpi slt, %arg4, %35 : i32
          scf.condition(%36) %arg4 : i32
        } do {
        ^bb0(%arg4: i32):
          %35 = arith.index_cast %arg4 : i32 to index
          %36 = memref.load %alloc_2[%35] : memref<?xmemref<?xi8>>
          memref.dealloc %36 : memref<?xi8>
          %37 = arith.addi %arg4, %c1_i32 : i32
          scf.yield %37 : i32
        }
        memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
        memref.dealloc %alloc : memref<?xf64>
        memref.dealloc %alloc_3 : memref<?x!llvm.struct<(i32, f64)>>
        %34 = "polygeist.pointer2memref"(%30) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, f64)>>
        scf.yield %34 : memref<?x!llvm.struct<(i32, f64)>>
      } else {
        affine.store %28, %alloca[0] : memref<1xmemref<?xi8>>
        %33 = llvm.mlir.addressof @str2 : !llvm.ptr
        %cast_4 = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %34 = "polygeist.pointer2memref"(%33) : (!llvm.ptr) -> memref<?xi8>
        %35 = llvm.mlir.zero : !llvm.ptr
        %36:5 = scf.while (%arg4 = %1, %arg5 = %0, %arg6 = %0, %arg7 = %0, %arg8 = %c0_i32, %arg9 = %true) : (f64, i32, i32, i32, i32, i1) -> (i32, f64, i32, i32, i32) {
          scf.condition(%arg9) %arg8, %arg4, %arg5, %arg6, %arg7 : i32, f64, i32, i32, i32
        } do {
        ^bb0(%arg4: i32, %arg5: f64, %arg6: i32, %arg7: i32, %arg8: i32):
          %44 = affine.load %alloca[0] : memref<1xmemref<?xi8>>
          %45 = func.call @strtok_r(%44, %34, %cast_4) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
          %46 = "polygeist.memref2pointer"(%45) : (memref<?xi8>) -> !llvm.ptr
          %47 = llvm.icmp "eq" %46, %35 : !llvm.ptr
          %48:6 = scf.if %47 -> (i32, i32, i1, i32, i32, f64) {
            scf.yield %arg7, %arg8, %false, %arg4, %arg6, %arg5 : i32, i32, i1, i32, i32, f64
          } else {
            %49 = affine.load %45[0] : memref<?xi8>
            %50 = arith.extui %49 : i8 to i32
            %51 = arith.cmpi ne, %50, %c0_i32 : i32
            %52:6 = scf.if %51 -> (i32, i32, i1, i32, i32, f64) {
              %53:2 = scf.while (%arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %true) : (i32, i32, i1) -> (i32, i32) {
                %62 = arith.cmpi slt, %arg9, %arg4 : i32
                %63 = arith.andi %62, %arg11 : i1
                scf.condition(%63) %arg9, %arg10 : i32, i32
              } do {
              ^bb0(%arg9: i32, %arg10: i32):
                %62 = arith.index_cast %arg9 : i32 to index
                %63 = memref.load %alloca_0[%62] : memref<256xmemref<?xi8>>
                %64 = func.call @compare_tokens(%45, %63) : (memref<?xi8>, memref<?xi8>) -> i32
                %65 = arith.cmpi eq, %64, %c0_i32 : i32
                %66 = arith.select %65, %c1_i32, %arg10 : i32
                %67 = arith.cmpi ne, %64, %c0_i32 : i32
                %68 = scf.if %67 -> (i32) {
                  %69 = arith.addi %arg9, %c1_i32 : i32
                  scf.yield %69 : i32
                } else {
                  scf.yield %arg9 : i32
                }
                scf.yield %68, %66, %67 : i32, i32, i1
              }
              %54 = arith.cmpi eq, %53#1, %c0_i32 : i32
              %55 = arith.cmpi slt, %arg4, %c256_i32 : i32
              %56 = arith.cmpi slt, %arg4, %c256_i32 : i32
              %57 = arith.andi %54, %55 : i1
              %58 = arith.cmpi ne, %53#1, %c0_i32 : i32
              %59 = arith.andi %54, %56 : i1
              %60 = arith.ori %59, %58 : i1
              %61:3 = scf.if %57 -> (i32, i32, f64) {
                %62 = arith.index_cast %arg4 : i32 to index
                %63 = func.call @strdup(%45) : (memref<?xi8>) -> memref<?xi8>
                memref.store %63, %alloca_0[%62] : memref<256xmemref<?xi8>>
                %64 = arith.index_cast %arg4 : i32 to index
                %65 = memref.load %alloca_0[%64] : memref<256xmemref<?xi8>>
                %66 = "polygeist.memref2pointer"(%65) : (memref<?xi8>) -> !llvm.ptr
                %67 = llvm.mlir.zero : !llvm.ptr
                %68 = llvm.icmp "eq" %66, %67 : !llvm.ptr
                %69:3 = scf.if %68 -> (i32, i32, f64) {
                  scf.yield %arg4, %arg6, %arg5 : i32, i32, f64
                } else {
                  %70 = arith.addi %arg4, %c1_i32 : i32
                  %71 = affine.load %alloca_1[0] : memref<1xi32>
                  %72 = func.call @calculate_df(%45, %alloc_2, %71) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32) -> i32
                  %73 = arith.cmpi ne, %72, %c0_i32 : i32
                  %74 = scf.if %73 -> (f64) {
                    %75 = affine.load %alloca_1[0] : memref<1xi32>
                    %76 = func.call @calculate_idf(%72, %75) : (i32, i32) -> f64
                    %77 = func.call @lower_dup(%45) : (memref<?xi8>) -> memref<?xi8>
                    %78 = "polygeist.memref2pointer"(%77) : (memref<?xi8>) -> !llvm.ptr
                    %79 = llvm.mlir.zero : !llvm.ptr
                    %80 = llvm.icmp "eq" %78, %79 : !llvm.ptr
                    %81 = arith.xori %80, %true : i1
                    scf.if %81 {
                      %82 = affine.load %alloca_1[0] : memref<1xi32>
                      %83 = arith.mulf %76, %cst : f64
                      %84 = arith.fptosi %83 : f64 to i32
                      func.call @score_term_over_docs(%77, %alloc_2, %alloc, %26, %76, %alloc_3, %82, %84) : (memref<?xi8>, memref<?xmemref<?xi8>>, memref<?xf64>, f64, f64, memref<?x!llvm.struct<(i32, f64)>>, i32, i32) -> ()
                      memref.dealloc %77 : memref<?xi8>
                    }
                    scf.yield %76 : f64
                  } else {
                    scf.yield %arg5 : f64
                  }
                  scf.yield %70, %72, %74 : i32, i32, f64
                }
                scf.yield %69#0, %69#1, %69#2 : i32, i32, f64
              } else {
                scf.yield %arg4, %arg6, %arg5 : i32, i32, f64
              }
              scf.yield %53#0, %53#1, %60, %61#0, %61#1, %61#2 : i32, i32, i1, i32, i32, f64
            } else {
              scf.yield %arg7, %arg8, %true, %arg4, %arg6, %arg5 : i32, i32, i1, i32, i32, f64
            }
            scf.yield %52#0, %52#1, %52#2, %52#3, %52#4, %52#5 : i32, i32, i1, i32, i32, f64
          }
          scf.yield %48#5, %48#4, %48#0, %48#1, %48#3, %48#2 : f64, i32, i32, i32, i32, i1
        }
        memref.dealloc %28 : memref<?xi8>
        %37 = arith.index_cast %36#0 : i32 to index
        scf.for %arg4 = %c0 to %37 step %c1 {
          %44 = memref.load %alloca_0[%arg4] : memref<256xmemref<?xi8>>
          memref.dealloc %44 : memref<?xi8>
        }
        %38:2 = scf.while (%arg4 = %c0_i32) : (i32) -> (i32, i32) {
          %44 = affine.load %alloca_1[0] : memref<1xi32>
          %45 = arith.cmpi slt, %arg4, %44 : i32
          scf.condition(%45) %44, %arg4 : i32, i32
        } do {
        ^bb0(%arg4: i32, %arg5: i32):
          %44 = arith.index_cast %arg5 : i32 to index
          %45 = memref.load %alloc_2[%44] : memref<?xmemref<?xi8>>
          memref.dealloc %45 : memref<?xi8>
          %46 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %46 : i32
        }
        memref.dealloc %alloc_2 : memref<?xmemref<?xi8>>
        memref.dealloc %alloc : memref<?xf64>
        %39 = "polygeist.memref2pointer"(%alloc_3) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
        %40 = "polygeist.pointer2memref"(%39) : (!llvm.ptr) -> memref<?xi8>
        %41 = arith.extsi %38#0 : i32 to i64
        %42 = "polygeist.get_func"() <{name = @compare_scores}> : () -> !llvm.ptr
        %43 = "polygeist.pointer2memref"(%42) : (!llvm.ptr) -> memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>
        func.call @qsort(%40, %41, %c16_i64, %43) : (memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) -> ()
        scf.yield %alloc_3 : memref<?x!llvm.struct<(i32, f64)>>
      }
      scf.yield %32 : memref<?x!llvm.struct<(i32, f64)>>
    }
    return %7 : memref<?x!llvm.struct<(i32, f64)>>
  }
  func.func private @strtok_r(memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @qsort(memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+06 : f64
    %cst_0 = arith.constant 1.000000e+03 : f64
    %c4_i32 = arith.constant 4 : i32
    %c6_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1x2xi64>
    %alloca_1 = memref.alloca() : memref<1x2xi64>
    %alloca_2 = memref.alloca() : memref<memref<?xmemref<?xi8>>>
    %alloca_3 = memref.alloca() : memref<1xi32>
    affine.store %0, %alloca_3[0] : memref<1xi32>
    %alloca_4 = memref.alloca() : memref<memref<?xi8>>
    %1 = arith.cmpi slt, %arg0, %c3_i32 : i32
    %2 = arith.cmpi sge, %arg0, %c3_i32 : i32
    %3 = arith.select %1, %c1_i32, %0 : i32
    scf.if %1 {
      %6 = llvm.mlir.addressof @stderr : !llvm.ptr
      %7 = llvm.load %6 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %8 = "polygeist.memref2pointer"(%7) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %9 = llvm.mlir.addressof @str3 : !llvm.ptr
      %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
      %11 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %12 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
      %13 = llvm.call @fprintf(%8, %10, %12) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %14 = llvm.mlir.addressof @stderr : !llvm.ptr
      %15 = llvm.load %14 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %16 = "polygeist.memref2pointer"(%15) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %17 = llvm.mlir.addressof @str4 : !llvm.ptr
      %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<60 x i8>
      %19 = llvm.call @fprintf(%16, %18) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %20 = llvm.mlir.addressof @stderr : !llvm.ptr
      %21 = llvm.load %20 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %22 = "polygeist.memref2pointer"(%21) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %23 = llvm.mlir.addressof @str5 : !llvm.ptr
      %24 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
      %25 = llvm.call @fprintf(%22, %24) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    %4:3 = scf.if %2 -> (i32, i1, i32) {
      %6 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %7 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
      affine.store %7, %alloca_4[] : memref<memref<?xi8>>
      %8 = func.call @rand() : () -> i32
      %9 = arith.remsi %8, %c6_i32 : i32
      %10 = arith.cmpi eq, %arg0, %c4_i32 : i32
      %11 = scf.if %10 -> (i32) {
        %26 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
        %27 = func.call @atoi(%26) : (memref<?xi8>) -> i32
        scf.yield %27 : i32
      } else {
        scf.yield %9 : i32
      }
      %12 = llvm.mlir.addressof @str6 : !llvm.ptr
      %13 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<13 x i8>
      %14 = "polygeist.memref2pointer"(%7) : (memref<?xi8>) -> !llvm.ptr
      %15 = llvm.call @printf(%13, %14) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %16 = llvm.mlir.addressof @str7 : !llvm.ptr
      %17 = llvm.getelementptr %16[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<29 x i8>
      %18 = "polygeist.memref2pointer"(%6) : (memref<?xi8>) -> !llvm.ptr
      %19 = llvm.call @printf(%17, %18) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %cast = memref.cast %alloca_3 : memref<1xi32> to memref<?xi32>
      %20 = func.call @read_documents_from_file(%6, %cast) : (memref<?xi8>, memref<?xi32>) -> memref<?xmemref<?xi8>>
      affine.store %20, %alloca_2[] : memref<memref<?xmemref<?xi8>>>
      %21 = "polygeist.memref2pointer"(%20) : (memref<?xmemref<?xi8>>) -> !llvm.ptr
      %22 = llvm.mlir.zero : !llvm.ptr
      %23 = llvm.icmp "eq" %21, %22 : !llvm.ptr
      %24 = arith.xori %23, %true : i1
      %25 = arith.select %23, %c1_i32, %3 : i32
      scf.if %23 {
        %26 = llvm.mlir.addressof @stderr : !llvm.ptr
        %27 = llvm.load %26 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
        %28 = "polygeist.memref2pointer"(%27) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
        %29 = llvm.mlir.addressof @str8 : !llvm.ptr
        %30 = llvm.getelementptr %29[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<37 x i8>
        %31 = llvm.call @fprintf(%28, %30) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      }
      scf.yield %11, %24, %25 : i32, i1, i32
    } else {
      scf.yield %0, %false, %3 : i32, i1, i32
    }
    %5 = arith.select %4#1, %c0_i32, %4#2 : i32
    scf.if %4#1 {
      %6 = llvm.mlir.addressof @str9 : !llvm.ptr
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
      %8 = affine.load %alloca_3[0] : memref<1xi32>
      %9 = llvm.call @printf(%7, %8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %cast = memref.cast %alloca_1 : memref<1x2xi64> to memref<?x2xi64>
      %10 = func.call @clock_gettime(%c1_i32, %cast) : (i32, memref<?x2xi64>) -> i32
      %11 = affine.load %alloca_4[] : memref<memref<?xi8>>
      %12 = affine.load %alloca_2[] : memref<memref<?xmemref<?xi8>>>
      %13 = affine.load %alloca_3[0] : memref<1xi32>
      %14 = func.call @rank_documents_bm25(%11, %12, %13, %4#0) : (memref<?xi8>, memref<?xmemref<?xi8>>, i32, i32) -> memref<?x!llvm.struct<(i32, f64)>>
      %cast_5 = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
      %15 = func.call @clock_gettime(%c1_i32, %cast_5) : (i32, memref<?x2xi64>) -> i32
      %16 = affine.load %alloca[0, 0] : memref<1x2xi64>
      %17 = affine.load %alloca_1[0, 0] : memref<1x2xi64>
      %18 = arith.subi %16, %17 : i64
      %19 = arith.sitofp %18 : i64 to f64
      %20 = arith.mulf %19, %cst_0 : f64
      %21 = affine.load %alloca[0, 1] : memref<1x2xi64>
      %22 = affine.load %alloca_1[0, 1] : memref<1x2xi64>
      %23 = arith.subi %21, %22 : i64
      %24 = arith.sitofp %23 : i64 to f64
      %25 = arith.divf %24, %cst : f64
      %26 = arith.addf %20, %25 : f64
      %27 = "polygeist.memref2pointer"(%14) : (memref<?x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
      %28 = llvm.mlir.zero : !llvm.ptr
      %29 = llvm.icmp "ne" %27, %28 : !llvm.ptr
      scf.if %29 {
        %33 = llvm.mlir.addressof @str10 : !llvm.ptr
        %34 = llvm.getelementptr %33[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<18 x i8>
        %35 = llvm.call @printf(%34) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        scf.for %arg2 = %c0 to %c1000 step %c1 {
          %36 = arith.index_cast %arg2 : index to i32
          %37 = arith.muli %arg2, %c16 : index
          %38 = arith.index_cast %37 : index to i64
          %39 = llvm.getelementptr %27[%38] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %40 = llvm.load %39 : !llvm.ptr -> i32
          %41 = llvm.mlir.addressof @str11 : !llvm.ptr
          %42 = llvm.getelementptr %41[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
          %43 = arith.addi %36, %c1_i32 : i32
          %44 = llvm.getelementptr %39[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
          %45 = llvm.load %44 : !llvm.ptr -> f64
          %46 = arith.index_cast %40 : i32 to index
          %47 = memref.load %12[%46] : memref<?xmemref<?xi8>>
          %48 = "polygeist.memref2pointer"(%47) : (memref<?xi8>) -> !llvm.ptr
          %49 = llvm.call @printf(%42, %43, %40, %45, %48) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, f64, !llvm.ptr) -> i32
        }
        memref.dealloc %14 : memref<?x!llvm.struct<(i32, f64)>>
      } else {
        %33 = llvm.mlir.addressof @str12 : !llvm.ptr
        %34 = llvm.getelementptr %33[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
        %35 = llvm.call @printf(%34) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      }
      %30 = llvm.mlir.addressof @str13 : !llvm.ptr
      %31 = llvm.getelementptr %30[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<28 x i8>
      %32 = llvm.call @printf(%31, %26) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    }
    return %5 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @clock_gettime(i32, memref<?x2xi64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}

