module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu", "polygeist.target-cpu" = "generic", "polygeist.target-features" = "+fp-armv8,+neon,+outline-atomics,+v8a,-fmv"} {
  llvm.mlir.global internal constant @str12("Elapsed %.3f ms\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Error: No documents loaded.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("Error: Failed to parse query embedding\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("[%s]\00") {addr_space = 0 : i32}
  llvm.func @sprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str8("confidence = %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("Document embeddings should be provided via stdin\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("Query embedding should be comma-separated values\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("Usage: %s <query_embedding> [top_k]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("Rank %d: Doc %d (Score: %.4f) - \22%s\22\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  memref.global @query_embedding : memref<384xf32> = uninitialized
  llvm.mlir.global internal constant @str3("Error: Failed to allocate memory for embedding string.\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stderr() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  llvm.func @fprintf(!llvm.ptr, !llvm.ptr, ...) -> i32
  memref.global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>> = uninitialized
  llvm.mlir.global internal constant @str2("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("|\00") {addr_space = 0 : i32}
  llvm.mlir.global external @stdin() {addr_space = 0 : i32} : memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
  memref.global @num_documents : memref<1xi32> = dense<0>
  llvm.mlir.global internal constant @str0(",]\00") {addr_space = 0 : i32}
  func.func @approx_state_identity(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    return %arg0 : i32
  }
  func.func @__internal_parse_embedding(%arg0: memref<?xi8>, %arg1: memref<?xf32>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c384_i32 = arith.constant 384 : i32
    %c0_i32 = arith.constant 0 : i32
    %c91_i32 = arith.constant 91 : i32
    %c1 = arith.constant 1 : index
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %0 = llvm.mlir.undef : i32
    %1 = call @strchr(%arg0, %c91_i32) : (memref<?xi8>, i32) -> memref<?xi8>
    %2 = "polygeist.memref2pointer"(%1) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.icmp "eq" %2, %3 : !llvm.ptr
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6 = scf.if %4 -> (i32) {
      scf.yield %5 : i32
    } else {
      %7 = "polygeist.subindex"(%1, %c1) : (memref<?xi8>, index) -> memref<?xi8>
      %8 = func.call @strdup(%7) : (memref<?xi8>) -> memref<?xi8>
      %9 = "polygeist.memref2pointer"(%8) : (memref<?xi8>) -> !llvm.ptr
      %10 = llvm.mlir.zero : !llvm.ptr
      %11 = llvm.icmp "eq" %9, %10 : !llvm.ptr
      %12 = scf.if %11 -> (i32) {
        scf.yield %5 : i32
      } else {
        %13 = llvm.mlir.zero : !llvm.ptr
        %14 = "polygeist.pointer2memref"(%13) : (!llvm.ptr) -> memref<?xi8>
        affine.store %14, %alloca[0] : memref<1xmemref<?xi8>>
        %15 = llvm.mlir.addressof @str0 : !llvm.ptr
        %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %16 = "polygeist.pointer2memref"(%15) : (!llvm.ptr) -> memref<?xi8>
        %17 = func.call @strtok_r(%8, %16, %cast) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
        %18 = llvm.mlir.zero : !llvm.ptr
        %19 = "polygeist.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi8>
        %20 = llvm.mlir.addressof @str0 : !llvm.ptr
        %cast_0 = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
        %22:2 = scf.while (%arg3 = %c0_i32, %arg4 = %17) : (i32, memref<?xi8>) -> (i32, memref<?xi8>) {
          %25 = "polygeist.memref2pointer"(%arg4) : (memref<?xi8>) -> !llvm.ptr
          %26 = llvm.icmp "ne" %25, %18 : !llvm.ptr
          %27 = arith.cmpi slt, %arg3, %c384_i32 : i32
          %28 = arith.andi %26, %27 : i1
          scf.condition(%28) %arg3, %arg4 : i32, memref<?xi8>
        } do {
        ^bb0(%arg3: i32, %arg4: memref<?xi8>):
          %25 = arith.index_cast %arg3 : i32 to index
          %26 = func.call @atof(%arg4) : (memref<?xi8>) -> f64
          %27 = arith.truncf %26 : f64 to f32
          memref.store %27, %arg1[%25] : memref<?xf32>
          %28 = arith.addi %arg3, %c1_i32 : i32
          %29 = func.call @strtok_r(%19, %21, %cast_0) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
          scf.yield %28, %29 : i32, memref<?xi8>
        }
        memref.dealloc %8 : memref<?xi8>
        %23 = arith.cmpi eq, %22#0, %c384_i32 : i32
        %24 = arith.extui %23 : i1 to i32
        scf.yield %24 : i32
      }
      scf.yield %12 : i32
    }
    return %6 : i32
  }
  func.func @parse_embedding(%arg0: memref<?xi8>, %arg1: memref<?xf32>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = call @approx_state_identity(%arg2) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c5_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = scf.index_switch %3 -> i32 
    case 0 {
      %5 = func.call @approx_parse_embedding_2(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xf32>, i32) -> i32
      scf.yield %5 : i32
    }
    case 1 {
      %5 = func.call @approx_parse_embedding_2(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xf32>, i32) -> i32
      scf.yield %5 : i32
    }
    default {
      %5 = func.call @approx_parse_embedding_2(%arg0, %arg1, %arg2) : (memref<?xi8>, memref<?xf32>, i32) -> i32
      scf.yield %5 : i32
    }
    return %4 : i32
  }
  func.func private @strchr(memref<?xi8>, i32) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strdup(memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strtok_r(memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atof(memref<?xi8>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @approx_parse_embedding_1(%arg0: memref<?xi8>, %arg1: memref<?xf32>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %c384_i32 = arith.constant 384 : i32
    %c0_i32 = arith.constant 0 : i32
    %c91_i32 = arith.constant 91 : i32
    %c1 = arith.constant 1 : index
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %0 = llvm.mlir.undef : i32
    %1 = call @strchr(%arg0, %c91_i32) : (memref<?xi8>, i32) -> memref<?xi8>
    %2 = "polygeist.memref2pointer"(%1) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.icmp "eq" %2, %3 : !llvm.ptr
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6 = scf.if %4 -> (i32) {
      scf.yield %5 : i32
    } else {
      %7 = "polygeist.subindex"(%1, %c1) : (memref<?xi8>, index) -> memref<?xi8>
      %8 = func.call @strdup(%7) : (memref<?xi8>) -> memref<?xi8>
      %9 = "polygeist.memref2pointer"(%8) : (memref<?xi8>) -> !llvm.ptr
      %10 = llvm.mlir.zero : !llvm.ptr
      %11 = llvm.icmp "eq" %9, %10 : !llvm.ptr
      %12 = scf.if %11 -> (i32) {
        scf.yield %5 : i32
      } else {
        %13 = llvm.mlir.zero : !llvm.ptr
        %14 = "polygeist.pointer2memref"(%13) : (!llvm.ptr) -> memref<?xi8>
        affine.store %14, %alloca[0] : memref<1xmemref<?xi8>>
        %15 = llvm.mlir.addressof @str0 : !llvm.ptr
        %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %16 = "polygeist.pointer2memref"(%15) : (!llvm.ptr) -> memref<?xi8>
        %17 = func.call @strtok_r(%8, %16, %cast) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
        %18 = llvm.mlir.zero : !llvm.ptr
        %19 = "polygeist.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi8>
        %20 = llvm.mlir.addressof @str0 : !llvm.ptr
        %cast_0 = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
        %22:2 = scf.while (%arg3 = %c0_i32, %arg4 = %17) : (i32, memref<?xi8>) -> (i32, memref<?xi8>) {
          %25 = "polygeist.memref2pointer"(%arg4) : (memref<?xi8>) -> !llvm.ptr
          %26 = llvm.icmp "ne" %25, %18 : !llvm.ptr
          %27 = arith.cmpi slt, %arg3, %c384_i32 : i32
          %28 = arith.andi %26, %27 : i1
          scf.condition(%28) %arg3, %arg4 : i32, memref<?xi8>
        } do {
        ^bb0(%arg3: i32, %arg4: memref<?xi8>):
          affine.store %c0_i8, %arg4[8] : memref<?xi8>
          %25 = arith.index_cast %arg3 : i32 to index
          %26 = func.call @atof(%arg4) : (memref<?xi8>) -> f64
          %27 = arith.truncf %26 : f64 to f32
          memref.store %27, %arg1[%25] : memref<?xf32>
          %28 = arith.addi %arg3, %c1_i32 : i32
          %29 = func.call @strtok_r(%19, %21, %cast_0) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
          scf.yield %28, %29 : i32, memref<?xi8>
        }
        memref.dealloc %8 : memref<?xi8>
        %23 = arith.cmpi eq, %22#0, %c384_i32 : i32
        %24 = arith.extui %23 : i1 to i32
        scf.yield %24 : i32
      }
      scf.yield %12 : i32
    }
    return %6 : i32
  }
  func.func @approx_parse_embedding_2(%arg0: memref<?xi8>, %arg1: memref<?xf32>, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %c384_i32 = arith.constant 384 : i32
    %c0_i32 = arith.constant 0 : i32
    %c91_i32 = arith.constant 91 : i32
    %c1 = arith.constant 1 : index
    %alloca = memref.alloca() : memref<1xmemref<?xi8>>
    %0 = llvm.mlir.undef : i32
    %1 = call @strchr(%arg0, %c91_i32) : (memref<?xi8>, i32) -> memref<?xi8>
    %2 = "polygeist.memref2pointer"(%1) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.icmp "eq" %2, %3 : !llvm.ptr
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6 = scf.if %4 -> (i32) {
      scf.yield %5 : i32
    } else {
      %7 = "polygeist.subindex"(%1, %c1) : (memref<?xi8>, index) -> memref<?xi8>
      %8 = func.call @strdup(%7) : (memref<?xi8>) -> memref<?xi8>
      %9 = "polygeist.memref2pointer"(%8) : (memref<?xi8>) -> !llvm.ptr
      %10 = llvm.mlir.zero : !llvm.ptr
      %11 = llvm.icmp "eq" %9, %10 : !llvm.ptr
      %12 = scf.if %11 -> (i32) {
        scf.yield %5 : i32
      } else {
        %13 = llvm.mlir.zero : !llvm.ptr
        %14 = "polygeist.pointer2memref"(%13) : (!llvm.ptr) -> memref<?xi8>
        affine.store %14, %alloca[0] : memref<1xmemref<?xi8>>
        %15 = llvm.mlir.addressof @str0 : !llvm.ptr
        %cast = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %16 = "polygeist.pointer2memref"(%15) : (!llvm.ptr) -> memref<?xi8>
        %17 = func.call @strtok_r(%8, %16, %cast) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
        %18 = llvm.mlir.zero : !llvm.ptr
        %19 = "polygeist.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi8>
        %20 = llvm.mlir.addressof @str0 : !llvm.ptr
        %cast_0 = memref.cast %alloca : memref<1xmemref<?xi8>> to memref<?xmemref<?xi8>>
        %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
        %22:2 = scf.while (%arg3 = %c0_i32, %arg4 = %17) : (i32, memref<?xi8>) -> (i32, memref<?xi8>) {
          %25 = "polygeist.memref2pointer"(%arg4) : (memref<?xi8>) -> !llvm.ptr
          %26 = llvm.icmp "ne" %25, %18 : !llvm.ptr
          %27 = arith.cmpi slt, %arg3, %c384_i32 : i32
          %28 = arith.andi %26, %27 : i1
          scf.condition(%28) %arg3, %arg4 : i32, memref<?xi8>
        } do {
        ^bb0(%arg3: i32, %arg4: memref<?xi8>):
          affine.store %c0_i8, %arg4[5] : memref<?xi8>
          %25 = arith.index_cast %arg3 : i32 to index
          %26 = func.call @atof(%arg4) : (memref<?xi8>) -> f64
          %27 = arith.truncf %26 : f64 to f32
          memref.store %27, %arg1[%25] : memref<?xf32>
          %28 = arith.addi %arg3, %c1_i32 : i32
          %29 = func.call @strtok_r(%19, %21, %cast_0) : (memref<?xi8>, memref<?xi8>, memref<?xmemref<?xi8>>) -> memref<?xi8>
          scf.yield %28, %29 : i32, memref<?xi8>
        }
        memref.dealloc %8 : memref<?xi8>
        %23 = arith.cmpi eq, %22#0, %c384_i32 : i32
        %24 = arith.extui %23 : i1 to i32
        scf.yield %24 : i32
      }
      scf.yield %12 : i32
    }
    return %6 : i32
  }
  func.func @load_document_embeddings() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %c255_i64 = arith.constant 255 : i64
    %c1_i64 = arith.constant 1 : i64
    %false = arith.constant false
    %c100000_i32 = arith.constant 100000 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<100000xi8>
    %0 = memref.get_global @num_documents : memref<1xi32>
    affine.store %c0_i32, %0[0] : memref<1xi32>
    %cast = memref.cast %alloca : memref<100000xi8> to memref<?xi8>
    %1 = llvm.mlir.addressof @stdin : !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    scf.while (%arg0 = %true) : (i1) -> () {
      %5 = llvm.load %1 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %6 = func.call @fgets(%cast, %c100000_i32, %5) : (memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8>
      %7 = "polygeist.memref2pointer"(%6) : (memref<?xi8>) -> !llvm.ptr
      %8 = llvm.icmp "ne" %7, %2 : !llvm.ptr
      %9 = scf.if %8 -> (i1) {
        %11 = memref.get_global @num_documents : memref<1xi32>
        %12 = affine.load %11[0] : memref<1xi32>
        %13 = arith.cmpi slt, %12, %c100000_i32 : i32
        scf.yield %13 : i1
      } else {
        scf.yield %false : i1
      }
      %10 = arith.andi %9, %arg0 : i1
      scf.condition(%10)
    } do {
      %5 = "polygeist.memref2pointer"(%alloca) : (memref<100000xi8>) -> !llvm.ptr
      %6 = func.call @strlen(%5) : (!llvm.ptr) -> i64
      %7 = arith.cmpi ugt, %6, %c1_i64 : i64
      %8 = scf.if %7 -> (i1) {
        %9 = llvm.mlir.addressof @str1 : !llvm.ptr
        %10 = "polygeist.pointer2memref"(%9) : (!llvm.ptr) -> memref<?xi8>
        %11 = func.call @strtok(%cast, %10) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
        %12 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi8>
        %13 = llvm.mlir.addressof @str1 : !llvm.ptr
        %14 = "polygeist.pointer2memref"(%13) : (!llvm.ptr) -> memref<?xi8>
        %15 = func.call @strtok(%12, %14) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
        %16 = "polygeist.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi8>
        %17 = llvm.mlir.addressof @str2 : !llvm.ptr
        %18 = "polygeist.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xi8>
        %19 = func.call @strtok(%16, %18) : (memref<?xi8>, memref<?xi8>) -> memref<?xi8>
        %20 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
        %21 = llvm.icmp "eq" %20, %2 : !llvm.ptr
        %22 = scf.if %21 -> (i1) {
          scf.yield %true : i1
        } else {
          %25 = "polygeist.memref2pointer"(%15) : (memref<?xi8>) -> !llvm.ptr
          %26 = llvm.icmp "eq" %25, %2 : !llvm.ptr
          scf.yield %26 : i1
        }
        %23 = scf.if %22 -> (i1) {
          scf.yield %true : i1
        } else {
          %25 = "polygeist.memref2pointer"(%19) : (memref<?xi8>) -> !llvm.ptr
          %26 = llvm.icmp "eq" %25, %2 : !llvm.ptr
          scf.yield %26 : i1
        }
        %24 = scf.if %23 -> (i1) {
          scf.yield %true : i1
        } else {
          %25 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
          %26 = memref.get_global @num_documents : memref<1xi32>
          %27 = affine.load %26[0] : memref<1xi32>
          %28 = arith.index_cast %27 : i32 to index
          %29 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
          %30 = arith.muli %28, %29 : index
          %31 = arith.index_cast %30 : index to i64
          %32 = "polygeist.memref2pointer"(%25) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
          %33 = llvm.getelementptr %32[%31] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %34 = func.call @atoi(%11) : (memref<?xi8>) -> i32
          llvm.store %34, %33 : i32, !llvm.ptr
          %35 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
          %36 = memref.get_global @num_documents : memref<1xi32>
          %37 = affine.load %36[0] : memref<1xi32>
          %38 = arith.index_cast %37 : i32 to index
          %39 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
          %40 = arith.muli %38, %39 : index
          %41 = arith.index_cast %40 : index to i64
          %42 = "polygeist.memref2pointer"(%35) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
          %43 = llvm.getelementptr %42[%41] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %44 = llvm.getelementptr %43[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
          %45 = "polygeist.pointer2memref"(%44) : (!llvm.ptr) -> memref<?xi8>
          %46 = func.call @strncpy(%45, %15, %c255_i64) : (memref<?xi8>, memref<?xi8>, i64) -> memref<?xi8>
          %47 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
          %48 = memref.get_global @num_documents : memref<1xi32>
          %49 = affine.load %48[0] : memref<1xi32>
          %50 = arith.index_cast %49 : i32 to index
          %51 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
          %52 = arith.muli %50, %51 : index
          %53 = arith.index_cast %52 : index to i64
          %54 = "polygeist.memref2pointer"(%47) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
          %55 = llvm.getelementptr %54[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %56 = llvm.getelementptr %55[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
          %57 = llvm.getelementptr %56[255] : (!llvm.ptr) -> !llvm.ptr, i8
          llvm.store %c0_i8, %57 : i8, !llvm.ptr
          %58 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
          %59 = memref.get_global @num_documents : memref<1xi32>
          %60 = affine.load %59[0] : memref<1xi32>
          %61 = arith.index_cast %60 : i32 to index
          %62 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
          %63 = arith.muli %61, %62 : index
          %64 = arith.index_cast %63 : index to i64
          %65 = "polygeist.memref2pointer"(%58) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
          %66 = llvm.getelementptr %65[%64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %67 = llvm.getelementptr %66[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
          %68 = func.call @strdup(%19) : (memref<?xi8>) -> memref<?xi8>
          llvm.store %68, %67 : memref<?xi8>, !llvm.ptr
          %69 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
          %70 = memref.get_global @num_documents : memref<1xi32>
          %71 = affine.load %70[0] : memref<1xi32>
          %72 = arith.index_cast %71 : i32 to index
          %73 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
          %74 = arith.muli %72, %73 : index
          %75 = arith.index_cast %74 : index to i64
          %76 = "polygeist.memref2pointer"(%69) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
          %77 = llvm.getelementptr %76[%75] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %78 = llvm.getelementptr %77[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
          %79 = llvm.load %78 : !llvm.ptr -> memref<?xi8>
          %80 = "polygeist.memref2pointer"(%79) : (memref<?xi8>) -> !llvm.ptr
          %81 = llvm.icmp "ne" %80, %2 : !llvm.ptr
          scf.if %81 {
            %82 = arith.addi %71, %c1_i32 : i32
            affine.store %82, %70[0] : memref<1xi32>
          } else {
            %82 = llvm.mlir.addressof @stderr : !llvm.ptr
            %83 = llvm.load %82 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
            %84 = "polygeist.memref2pointer"(%83) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
            %85 = llvm.mlir.addressof @str3 : !llvm.ptr
            %86 = llvm.getelementptr %85[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<56 x i8>
            %87 = llvm.call @fprintf(%84, %86) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
          }
          scf.yield %81 : i1
        }
        scf.yield %24 : i1
      } else {
        scf.yield %true : i1
      }
      scf.yield %8 : i1
    }
    %3 = memref.get_global @num_documents : memref<1xi32>
    %4 = affine.load %3[0] : memref<1xi32>
    return %4 : i32
  }
  func.func private @fgets(memref<?xi8>, i32, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strlen(!llvm.ptr) -> i64
  func.func private @strtok(memref<?xi8>, memref<?xi8>) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @strncpy(memref<?xi8>, memref<?xi8>, i64) -> memref<?xi8> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @cosine_similarity_core(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c384 = arith.constant 384 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f32
    %0:3 = scf.for %arg2 = %c0 to %c384 step %c1 iter_args(%arg3 = %cst, %arg4 = %cst, %arg5 = %cst) -> (f32, f32, f32) {
      %6 = memref.load %arg0[%arg2] : memref<?xf32>
      %7 = memref.load %arg1[%arg2] : memref<?xf32>
      %8 = arith.mulf %6, %7 : f32
      %9 = arith.addf %arg5, %8 : f32
      %10 = arith.mulf %6, %6 : f32
      %11 = arith.addf %arg4, %10 : f32
      %12 = arith.mulf %7, %7 : f32
      %13 = arith.addf %arg3, %12 : f32
      scf.yield %13, %11, %9 : f32, f32, f32
    }
    %1 = math.sqrt %0#1 : f32
    %2 = math.sqrt %0#0 : f32
    %3 = arith.cmpf oeq, %1, %cst : f32
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %6 = arith.cmpf oeq, %2, %cst : f32
      scf.yield %6 : i1
    }
    %5 = scf.if %4 -> (f32) {
      scf.yield %cst : f32
    } else {
      %6 = arith.mulf %1, %2 : f32
      %7 = arith.divf %0#2, %6 : f32
      scf.yield %7 : f32
    }
    return %5 : f32
  }
  func.func @approx_compute_similarities_with_state_1(%arg0: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -2.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c95_i32 = arith.constant 95 : i32
    %c100_i32 = arith.constant 100 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<384xf32>
    %0 = memref.get_global @num_documents : memref<1xi32>
    %1 = scf.while (%arg1 = %c0_i32) : (i32) -> i32 {
      %2 = affine.load %0[0] : memref<1xi32>
      %3 = arith.cmpi slt, %arg1, %2 : i32
      scf.condition(%3) %arg1 : i32
    } do {
    ^bb0(%arg1: i32):
      %2 = func.call @rand() : () -> i32
      %3 = arith.remsi %2, %c100_i32 : i32
      %4 = arith.cmpi slt, %3, %c95_i32 : i32
      scf.if %4 {
        %6 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
        %7 = arith.index_cast %arg1 : i32 to index
        %8 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
        %9 = arith.muli %7, %8 : index
        %10 = arith.index_cast %9 : index to i64
        %11 = "polygeist.memref2pointer"(%6) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
        %12 = llvm.getelementptr %11[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %13 = llvm.getelementptr %12[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
        %14 = llvm.load %13 : !llvm.ptr -> memref<?xi8>
        %cast = memref.cast %alloca : memref<384xf32> to memref<?xf32>
        %15 = func.call @rand() : () -> i32
        %16 = arith.remsi %15, %c10_i32 : i32
        %17 = func.call @parse_embedding(%14, %cast, %16) : (memref<?xi8>, memref<?xf32>, i32) -> i32
        %18 = arith.cmpi ne, %17, %c0_i32 : i32
        scf.if %18 {
          %19 = llvm.getelementptr %12[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
          %20 = memref.get_global @query_embedding : memref<384xf32>
          %cast_0 = memref.cast %20 : memref<384xf32> to memref<?xf32>
          %21 = func.call @cosine_similarity_core(%cast_0, %cast) : (memref<?xf32>, memref<?xf32>) -> f32
          llvm.store %21, %19 : f32, !llvm.ptr
        } else {
          func.call @exit(%c-1_i32) : (i32) -> ()
        }
      } else {
        %6 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
        %7 = arith.index_cast %arg1 : i32 to index
        %8 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
        %9 = arith.muli %7, %8 : index
        %10 = arith.index_cast %9 : index to i64
        %11 = "polygeist.memref2pointer"(%6) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
        %12 = llvm.getelementptr %11[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %13 = llvm.getelementptr %12[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
        llvm.store %cst, %13 : f32, !llvm.ptr
      }
      %5 = arith.addi %arg1, %c1_i32 : i32
      scf.yield %5 : i32
    }
    return
  }
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @exit(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @approx_compute_similarities_with_state_2(%arg0: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -2.000000e+00 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c90_i32 = arith.constant 90 : i32
    %c100_i32 = arith.constant 100 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<384xf32>
    %0 = memref.get_global @num_documents : memref<1xi32>
    %1 = scf.while (%arg1 = %c0_i32) : (i32) -> i32 {
      %2 = affine.load %0[0] : memref<1xi32>
      %3 = arith.cmpi slt, %arg1, %2 : i32
      scf.condition(%3) %arg1 : i32
    } do {
    ^bb0(%arg1: i32):
      %2 = func.call @rand() : () -> i32
      %3 = arith.remsi %2, %c100_i32 : i32
      %4 = arith.cmpi slt, %3, %c90_i32 : i32
      scf.if %4 {
        %6 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
        %7 = arith.index_cast %arg1 : i32 to index
        %8 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
        %9 = arith.muli %7, %8 : index
        %10 = arith.index_cast %9 : index to i64
        %11 = "polygeist.memref2pointer"(%6) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
        %12 = llvm.getelementptr %11[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %13 = llvm.getelementptr %12[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
        %14 = llvm.load %13 : !llvm.ptr -> memref<?xi8>
        %cast = memref.cast %alloca : memref<384xf32> to memref<?xf32>
        %15 = func.call @rand() : () -> i32
        %16 = arith.remsi %15, %c10_i32 : i32
        %17 = func.call @parse_embedding(%14, %cast, %16) : (memref<?xi8>, memref<?xf32>, i32) -> i32
        %18 = arith.cmpi ne, %17, %c0_i32 : i32
        scf.if %18 {
          %19 = llvm.getelementptr %12[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
          %20 = memref.get_global @query_embedding : memref<384xf32>
          %cast_0 = memref.cast %20 : memref<384xf32> to memref<?xf32>
          %21 = func.call @cosine_similarity_core(%cast_0, %cast) : (memref<?xf32>, memref<?xf32>) -> f32
          llvm.store %21, %19 : f32, !llvm.ptr
        } else {
          func.call @exit(%c-1_i32) : (i32) -> ()
        }
      } else {
        %6 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
        %7 = arith.index_cast %arg1 : i32 to index
        %8 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
        %9 = arith.muli %7, %8 : index
        %10 = arith.index_cast %9 : index to i64
        %11 = "polygeist.memref2pointer"(%6) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
        %12 = llvm.getelementptr %11[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %13 = llvm.getelementptr %12[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
        llvm.store %cst, %13 : f32, !llvm.ptr
      }
      %5 = arith.addi %arg1, %c1_i32 : i32
      scf.yield %5 : i32
    }
    return
  }
  func.func @__internal_compute_similarities_with_state(%arg0: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<384xf32>
    %0 = memref.get_global @num_documents : memref<1xi32>
    %1 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
    %2 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
    %3 = "polygeist.memref2pointer"(%1) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
    %cast = memref.cast %alloca : memref<384xf32> to memref<?xf32>
    %4 = scf.while (%arg1 = %c0_i32) : (i32) -> i32 {
      %5 = affine.load %0[0] : memref<1xi32>
      %6 = arith.cmpi slt, %arg1, %5 : i32
      scf.condition(%6) %arg1 : i32
    } do {
    ^bb0(%arg1: i32):
      %5 = arith.index_cast %arg1 : i32 to index
      %6 = arith.muli %5, %2 : index
      %7 = arith.index_cast %6 : index to i64
      %8 = llvm.getelementptr %3[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %9 = llvm.getelementptr %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
      %10 = llvm.load %9 : !llvm.ptr -> memref<?xi8>
      %11 = func.call @rand() : () -> i32
      %12 = arith.remsi %11, %c10_i32 : i32
      %13 = func.call @parse_embedding(%10, %cast, %12) : (memref<?xi8>, memref<?xf32>, i32) -> i32
      %14 = arith.cmpi ne, %13, %c0_i32 : i32
      scf.if %14 {
        %16 = llvm.getelementptr %8[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
        %17 = memref.get_global @query_embedding : memref<384xf32>
        %cast_0 = memref.cast %17 : memref<384xf32> to memref<?xf32>
        %18 = func.call @cosine_similarity_core(%cast_0, %cast) : (memref<?xf32>, memref<?xf32>) -> f32
        llvm.store %18, %16 : f32, !llvm.ptr
      } else {
        func.call @exit(%c-1_i32) : (i32) -> ()
      }
      %15 = arith.addi %arg1, %c1_i32 : i32
      scf.yield %15 : i32
    }
    return
  }
  func.func @compute_similarities_with_state(%arg0: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = call @approx_state_identity(%arg0) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c5_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.index_switch %3 
    case 0 {
      func.call @approx_compute_similarities_with_state_1(%arg0) : (i32) -> ()
      scf.yield
    }
    case 1 {
      func.call @approx_compute_similarities_with_state_1(%arg0) : (i32) -> ()
      scf.yield
    }
    default {
      func.call @approx_compute_similarities_with_state_1(%arg0) : (i32) -> ()
    }
    return
  }
  func.func @compare_docs_desc(%arg0: memref<?xi8>, %arg1: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = "polygeist.memref2pointer"(%arg1) : (memref<?xi8>) -> !llvm.ptr
    %3 = llvm.getelementptr %1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
    %4 = llvm.load %3 : !llvm.ptr -> f32
    %5 = llvm.getelementptr %2[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
    %6 = llvm.load %5 : !llvm.ptr -> f32
    %7 = arith.cmpf ogt, %4, %6 : f32
    %8 = arith.select %7, %c-1_i32, %0 : i32
    %9:2 = scf.if %7 -> (i1, i32) {
      scf.yield %false, %8 : i1, i32
    } else {
      %11 = llvm.getelementptr %1[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
      %12 = llvm.load %11 : !llvm.ptr -> f32
      %13 = llvm.getelementptr %2[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
      %14 = llvm.load %13 : !llvm.ptr -> f32
      %15 = arith.cmpf olt, %12, %14 : f32
      %16 = arith.xori %15, %true : i1
      %17 = arith.select %15, %c1_i32, %8 : i32
      scf.yield %16, %17 : i1, i32
    }
    %10 = arith.select %9#0, %c0_i32, %9#1 : i32
    return %10 : i32
  }
  func.func @rank_topk(%arg0: i32, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
    %1 = "polygeist.memref2pointer"(%0) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi8>
    %3 = memref.get_global @num_documents : memref<1xi32>
    %4 = affine.load %3[0] : memref<1xi32>
    %5 = arith.extsi %4 : i32 to i64
    %6 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
    %7 = arith.index_cast %6 : index to i64
    %8 = "polygeist.get_func"() <{name = @compare_docs_desc}> : () -> !llvm.ptr
    %9 = "polygeist.pointer2memref"(%8) : (!llvm.ptr) -> memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>
    call @qsort(%2, %5, %7, %9) : (memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) -> ()
    return
  }
  func.func private @qsort(memref<?xi8>, i64, i64, memref<?x!llvm.func<i32 (memref<?xi8>, memref<?xi8>)>>) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @output_ranked_docs() attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @num_documents : memref<1xi32>
    %1 = llvm.mlir.addressof @str4 : !llvm.ptr
    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<38 x i8>
    %3 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
    %4 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
    %5 = "polygeist.memref2pointer"(%3) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
    %6 = scf.while (%arg0 = %c0_i32) : (i32) -> i32 {
      %7 = affine.load %0[0] : memref<1xi32>
      %8 = arith.cmpi slt, %arg0, %7 : i32
      scf.condition(%8) %arg0 : i32
    } do {
    ^bb0(%arg0: i32):
      %7 = arith.addi %arg0, %c1_i32 : i32
      %8 = arith.index_cast %arg0 : i32 to index
      %9 = arith.muli %8, %4 : index
      %10 = arith.index_cast %9 : index to i64
      %11 = llvm.getelementptr %5[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %12 = llvm.load %11 : !llvm.ptr -> i32
      %13 = llvm.getelementptr %11[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
      %14 = llvm.load %13 : !llvm.ptr -> f32
      %15 = arith.extf %14 : f32 to f64
      %16 = llvm.getelementptr %11[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
      %17 = llvm.call @printf(%2, %7, %12, %15, %16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, f64, !llvm.ptr) -> i32
      %18 = arith.addi %arg0, %c1_i32 : i32
      scf.yield %18 : i32
    }
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %cst = arith.constant 1.000000e+06 : f64
    %cst_0 = arith.constant 1.000000e+03 : f64
    %c0_i32 = arith.constant 0 : i32
    %c3_i64 = arith.constant 3 : i64
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1x2xi64>
    %alloca_1 = memref.alloca() : memref<1x2xi64>
    %1 = arith.cmpi slt, %arg0, %c2_i32 : i32
    %2 = arith.cmpi sge, %arg0, %c2_i32 : i32
    %3 = arith.select %1, %c1_i32, %0 : i32
    scf.if %1 {
      %6 = llvm.mlir.addressof @stderr : !llvm.ptr
      %7 = llvm.load %6 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %8 = "polygeist.memref2pointer"(%7) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %9 = llvm.mlir.addressof @str5 : !llvm.ptr
      %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<37 x i8>
      %11 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
      %12 = "polygeist.memref2pointer"(%11) : (memref<?xi8>) -> !llvm.ptr
      %13 = llvm.call @fprintf(%8, %10, %12) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %14 = llvm.mlir.addressof @stderr : !llvm.ptr
      %15 = llvm.load %14 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %16 = "polygeist.memref2pointer"(%15) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %17 = llvm.mlir.addressof @str6 : !llvm.ptr
      %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<50 x i8>
      %19 = llvm.call @fprintf(%16, %18) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      %20 = llvm.mlir.addressof @stderr : !llvm.ptr
      %21 = llvm.load %20 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %22 = "polygeist.memref2pointer"(%21) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %23 = llvm.mlir.addressof @str7 : !llvm.ptr
      %24 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<50 x i8>
      %25 = llvm.call @fprintf(%22, %24) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
    }
    %4:4 = scf.if %2 -> (i32, i32, i1, i32) {
      %6 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %7 = arith.cmpi sgt, %arg0, %c2_i32 : i32
      %8 = scf.if %7 -> (i32) {
        %29 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %30 = func.call @atoi(%29) : (memref<?xi8>) -> i32
        scf.yield %30 : i32
      } else {
        scf.yield %c10_i32 : i32
      }
      %9 = arith.cmpi sgt, %arg0, %c3_i32 : i32
      %10 = scf.if %9 -> (i32) {
        %29 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
        %30 = func.call @atoi(%29) : (memref<?xi8>) -> i32
        scf.yield %30 : i32
      } else {
        %29 = func.call @rand() : () -> i32
        %30 = arith.remsi %29, %c5_i32 : i32
        %31 = arith.addi %30, %c1_i32 : i32
        scf.yield %31 : i32
      }
      %11 = llvm.mlir.addressof @str8 : !llvm.ptr
      %12 = llvm.getelementptr %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
      %13 = llvm.call @printf(%12, %10) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %14 = "polygeist.memref2pointer"(%6) : (memref<?xi8>) -> !llvm.ptr
      %15 = func.call @strlen(%14) : (!llvm.ptr) -> i64
      %16 = arith.addi %15, %c3_i64 : i64
      %17 = arith.index_cast %16 : i64 to index
      %alloca_2 = memref.alloca(%17) : memref<?xi8>
      %18 = "polygeist.memref2pointer"(%alloca_2) : (memref<?xi8>) -> !llvm.ptr
      %19 = llvm.mlir.addressof @str9 : !llvm.ptr
      %20 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
      %21 = "polygeist.memref2pointer"(%6) : (memref<?xi8>) -> !llvm.ptr
      %22 = llvm.call @sprintf(%18, %20, %21) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %23 = memref.get_global @query_embedding : memref<384xf32>
      %cast = memref.cast %23 : memref<384xf32> to memref<?xf32>
      %24 = func.call @parse_embedding(%alloca_2, %cast, %c0_i32) : (memref<?xi8>, memref<?xf32>, i32) -> i32
      %25 = arith.cmpi eq, %24, %c0_i32 : i32
      %26 = arith.cmpi ne, %24, %c0_i32 : i32
      %27 = arith.select %25, %c1_i32, %3 : i32
      scf.if %25 {
        %29 = llvm.mlir.addressof @stderr : !llvm.ptr
        %30 = llvm.load %29 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
        %31 = "polygeist.memref2pointer"(%30) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
        %32 = llvm.mlir.addressof @str10 : !llvm.ptr
        %33 = llvm.getelementptr %32[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<40 x i8>
        %34 = llvm.call @fprintf(%31, %33) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
      }
      %28:2 = scf.if %26 -> (i1, i32) {
        %29 = func.call @load_document_embeddings() : () -> i32
        %30 = arith.cmpi eq, %29, %c0_i32 : i32
        %31 = arith.cmpi ne, %29, %c0_i32 : i32
        %32 = arith.select %30, %c1_i32, %27 : i32
        scf.if %30 {
          %33 = llvm.mlir.addressof @stderr : !llvm.ptr
          %34 = llvm.load %33 : !llvm.ptr -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
          %35 = "polygeist.memref2pointer"(%34) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
          %36 = llvm.mlir.addressof @str11 : !llvm.ptr
          %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<29 x i8>
          %38 = llvm.call @fprintf(%35, %37) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        }
        scf.yield %31, %32 : i1, i32
      } else {
        scf.yield %false, %3 : i1, i32
      }
      scf.yield %8, %10, %28#0, %28#1 : i32, i32, i1, i32
    } else {
      scf.yield %0, %0, %false, %3 : i32, i32, i1, i32
    }
    %5 = arith.select %4#2, %c0_i32, %4#3 : i32
    scf.if %4#2 {
      %cast = memref.cast %alloca_1 : memref<1x2xi64> to memref<?x2xi64>
      %6 = func.call @clock_gettime(%c1_i32, %cast) : (i32, memref<?x2xi64>) -> i32
      func.call @compute_similarities_with_state(%4#1) : (i32) -> ()
      func.call @rank_topk(%4#0, %4#1) : (i32, i32) -> ()
      %cast_2 = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
      %7 = func.call @clock_gettime(%c1_i32, %cast_2) : (i32, memref<?x2xi64>) -> i32
      %8 = affine.load %alloca[0, 0] : memref<1x2xi64>
      %9 = affine.load %alloca_1[0, 0] : memref<1x2xi64>
      %10 = arith.subi %8, %9 : i64
      %11 = arith.sitofp %10 : i64 to f64
      %12 = arith.mulf %11, %cst_0 : f64
      %13 = affine.load %alloca[0, 1] : memref<1x2xi64>
      %14 = affine.load %alloca_1[0, 1] : memref<1x2xi64>
      %15 = arith.subi %13, %14 : i64
      %16 = arith.sitofp %15 : i64 to f64
      %17 = arith.divf %16, %cst : f64
      %18 = arith.addf %12, %17 : f64
      %19 = llvm.mlir.addressof @str12 : !llvm.ptr
      %20 = llvm.getelementptr %19[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
      %21 = llvm.call @printf(%20, %18) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      func.call @output_ranked_docs() : () -> ()
      %22 = memref.get_global @num_documents : memref<1xi32>
      %23 = memref.get_global @documents : memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>
      %24 = "polygeist.typeSize"() <{source = !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>}> : () -> index
      %25 = "polygeist.memref2pointer"(%23) : (memref<100000x!llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>>) -> !llvm.ptr
      %26 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %27 = affine.load %22[0] : memref<1xi32>
        %28 = arith.cmpi slt, %arg2, %27 : i32
        scf.condition(%28) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        %27 = arith.index_cast %arg2 : i32 to index
        %28 = arith.muli %27, %24 : index
        %29 = arith.index_cast %28 : index to i64
        %30 = llvm.getelementptr %25[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %31 = llvm.getelementptr %30[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, memref<?xi8>, array<256 x i8>, f32)>
        %32 = llvm.load %31 : !llvm.ptr -> memref<?xi8>
        memref.dealloc %32 : memref<?xi8>
        %33 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %33 : i32
      }
    }
    return %5 : i32
  }
  func.func private @clock_gettime(i32, memref<?x2xi64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}

