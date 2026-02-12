module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi32>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu", "polygeist.target-cpu" = "generic", "polygeist.target-features" = "+fp-armv8,+neon,+outline-atomics,+v8a,-fmv"} {
  llvm.mlir.global internal constant @str28("ERROR: open result.txt\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str27("Results written to result.txt\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str26("ERROR: data write\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str25("ERROR: header write\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str24("wb\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str23("result.txt\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str22("%14zu | %13.6f | %11.6f | %11.6f | %11.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str21("---------------|---------------|-------------|-------------|-------------\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str20("Particle Index | Potential (v) |   Force (x) |   Force (y) |   Force (z)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str19("\0A--- Full Particle Data Dump (fv_cpu) ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str18("      Force Vector (z) | %12.6f | %12.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("      Force Vector (y) | %12.6f | %12.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("      Force Vector (x) | %12.6f | %12.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("Potential Energy (v) | %12.6f | %12.6f | %12.6f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("------------------|--------------|--------------|--------------\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("        Component |      Average |          Min |          Max\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("\0A--- Result Summary (fv_cpu) ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("Particles per Box: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("Number of Boxes: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("Total Particles: %ld\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("\0A--- Simulation Statistics ---\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("Total execution time: %.3f ms\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("ERROR: OOM\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("Configuration: boxes1d = %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4("Usage: %s -boxes1d <number>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3("ERROR: Usage: %s -boxes1d <number>\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("ERROR: -boxes1d > 0\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("-boxes1d\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("WG size of kernel = %d\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @approx_state_identity(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    return %arg0 : i32
  }
  func.func @pair_interaction(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = affine.load %arg3[symbol(%0), 0] : memref<?x4xf64>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = affine.load %arg3[symbol(%2), 0] : memref<?x4xf64>
    %4 = arith.addf %1, %3 : f64
    %5 = affine.load %arg3[symbol(%0), 1] : memref<?x4xf64>
    %6 = affine.load %arg3[symbol(%2), 1] : memref<?x4xf64>
    %7 = arith.mulf %5, %6 : f64
    %8 = affine.load %arg3[symbol(%0), 2] : memref<?x4xf64>
    %9 = affine.load %arg3[symbol(%2), 2] : memref<?x4xf64>
    %10 = arith.mulf %8, %9 : f64
    %11 = arith.addf %7, %10 : f64
    %12 = affine.load %arg3[symbol(%0), 3] : memref<?x4xf64>
    %13 = affine.load %arg3[symbol(%2), 3] : memref<?x4xf64>
    %14 = arith.mulf %12, %13 : f64
    %15 = arith.addf %11, %14 : f64
    %16 = arith.subf %4, %15 : f64
    %17 = arith.cmpf olt, %16, %cst_0 : f64
    %18 = arith.select %17, %cst_0, %16 : f64
    %19 = arith.mulf %arg2, %18 : f64
    %20 = arith.negf %19 : f64
    %21 = math.exp %20 : f64
    %22 = arith.mulf %21, %cst : f64
    %23 = arith.index_cast %arg1 : i32 to index
    %24 = affine.load %arg4[symbol(%23)] : memref<?xf64>
    %25 = arith.mulf %24, %21 : f64
    %26 = affine.load %arg5[0, 0] : memref<?x4xf64>
    %27 = arith.addf %26, %25 : f64
    affine.store %27, %arg5[0, 0] : memref<?x4xf64>
    %28 = arith.index_cast %arg1 : i32 to index
    %29 = affine.load %arg4[symbol(%28)] : memref<?xf64>
    %30 = arith.index_cast %arg0 : i32 to index
    %31 = affine.load %arg3[symbol(%30), 1] : memref<?x4xf64>
    %32 = affine.load %arg3[symbol(%28), 1] : memref<?x4xf64>
    %33 = arith.subf %31, %32 : f64
    %34 = arith.mulf %22, %33 : f64
    %35 = arith.mulf %29, %34 : f64
    %36 = affine.load %arg5[0, 1] : memref<?x4xf64>
    %37 = arith.addf %36, %35 : f64
    affine.store %37, %arg5[0, 1] : memref<?x4xf64>
    %38 = arith.index_cast %arg1 : i32 to index
    %39 = affine.load %arg4[symbol(%38)] : memref<?xf64>
    %40 = arith.index_cast %arg0 : i32 to index
    %41 = affine.load %arg3[symbol(%40), 2] : memref<?x4xf64>
    %42 = affine.load %arg3[symbol(%38), 2] : memref<?x4xf64>
    %43 = arith.subf %41, %42 : f64
    %44 = arith.mulf %22, %43 : f64
    %45 = arith.mulf %39, %44 : f64
    %46 = affine.load %arg5[0, 2] : memref<?x4xf64>
    %47 = arith.addf %46, %45 : f64
    affine.store %47, %arg5[0, 2] : memref<?x4xf64>
    %48 = arith.index_cast %arg1 : i32 to index
    %49 = affine.load %arg4[symbol(%48)] : memref<?xf64>
    %50 = arith.index_cast %arg0 : i32 to index
    %51 = affine.load %arg3[symbol(%50), 3] : memref<?x4xf64>
    %52 = affine.load %arg3[symbol(%48), 3] : memref<?x4xf64>
    %53 = arith.subf %51, %52 : f64
    %54 = arith.mulf %22, %53 : f64
    %55 = arith.mulf %49, %54 : f64
    %56 = affine.load %arg5[0, 3] : memref<?x4xf64>
    %57 = arith.addf %56, %55 : f64
    affine.store %57, %arg5[0, 3] : memref<?x4xf64>
    return %c0_i32 : i32
  }
  func.func @__internal_neighbor_box_accumulate(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg4: memref<?x4xf64>, %arg5: memref<?xf64>, %arg6: memref<?x4xf64>, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c656 = arith.constant 656 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.muli %0, %c656 : index
    %2 = arith.index_cast %1 : index to i64
    %3 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
    %4 = llvm.getelementptr %3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %5 = llvm.getelementptr %4[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %6 = llvm.getelementptr %4[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %7 = scf.while (%arg8 = %c0_i32) : (i32) -> i32 {
      %8 = llvm.load %5 : !llvm.ptr -> i32
      %9 = arith.cmpi slt, %arg8, %8 : i32
      scf.condition(%9) %arg8 : i32
    } do {
    ^bb0(%arg8: i32):
      %8 = arith.index_cast %arg8 : i32 to index
      %9 = arith.muli %8, %c24 : index
      %10 = arith.index_cast %9 : index to i64
      %11 = llvm.getelementptr %6[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %12 = llvm.getelementptr %11[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
      %13 = llvm.load %12 : !llvm.ptr -> i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = arith.muli %14, %c656 : index
      %16 = arith.index_cast %15 : index to i64
      %17 = llvm.getelementptr %3[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %18 = llvm.getelementptr %17[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %19 = llvm.load %18 : !llvm.ptr -> i64
      %20 = arith.trunci %19 : i64 to i32
      scf.for %arg9 = %c0 to %c128 step %c1 {
        %22 = arith.index_cast %arg9 : index to i32
        %23 = arith.addi %20, %22 : i32
        %24 = func.call @pair_interaction(%arg0, %23, %arg2, %arg4, %arg5, %arg6) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
      }
      %21 = arith.addi %arg8, %c1_i32 : i32
      scf.yield %21 : i32
    }
    return
  }
  func.func @neighbor_box_accumulate(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg4: memref<?x4xf64>, %arg5: memref<?xf64>, %arg6: memref<?x4xf64>, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c6_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = call @approx_state_identity(%arg7) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c6_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.index_switch %3 
    case 0 {
      func.call @__internal_neighbor_box_accumulate(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (i32, i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> ()
      scf.yield
    }
    case 1 {
      func.call @__internal_neighbor_box_accumulate(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (i32, i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> ()
      scf.yield
    }
    default {
      func.call @__internal_neighbor_box_accumulate(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (i32, i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> ()
    }
    return
  }
  func.func @approx_neighbor_box_accumulate_1(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg4: memref<?x4xf64>, %arg5: memref<?xf64>, %arg6: memref<?x4xf64>, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c656 = arith.constant 656 : index
    %c24 = arith.constant 24 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f64
    %2 = llvm.mlir.undef : i32
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.muli %3, %c656 : index
    %5 = arith.index_cast %4 : index to i64
    %6 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
    %7 = llvm.getelementptr %6[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %8 = llvm.getelementptr %7[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %9 = llvm.getelementptr %7[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %10:3 = scf.while (%arg8 = %1, %arg9 = %2, %arg10 = %c0_i32) : (f64, i32, i32) -> (f64, i32, i32) {
      %11 = llvm.load %8 : !llvm.ptr -> i32
      %12 = arith.cmpi slt, %arg10, %11 : i32
      scf.condition(%12) %arg8, %arg9, %arg10 : f64, i32, i32
    } do {
    ^bb0(%arg8: f64, %arg9: i32, %arg10: i32):
      %11 = arith.index_cast %arg10 : i32 to index
      %12 = arith.muli %11, %c24 : index
      %13 = arith.index_cast %12 : index to i64
      %14 = llvm.getelementptr %9[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %15 = llvm.getelementptr %14[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
      %16 = llvm.load %15 : !llvm.ptr -> i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.muli %17, %c656 : index
      %19 = arith.index_cast %18 : index to i64
      %20 = llvm.getelementptr %6[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %21 = llvm.getelementptr %20[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %22 = llvm.load %21 : !llvm.ptr -> i64
      %23 = arith.trunci %22 : i64 to i32
      %24:2 = scf.for %arg11 = %c0 to %c128 step %c1 iter_args(%arg12 = %arg8, %arg13 = %arg9) -> (f64, i32) {
        %26 = arith.index_cast %arg11 : index to i32
        %27 = arith.andi %26, %c7_i32 : i32
        %28 = arith.cmpi ne, %27, %c7_i32 : i32
        %29:2 = scf.if %28 -> (i32, f64) {
          %30 = arith.addi %23, %26 : i32
          %31 = affine.load %arg4[symbol(%0), 0] : memref<?x4xf64>
          %32 = arith.index_cast %30 : i32 to index
          %33 = memref.load %arg4[%32, %c0] : memref<?x4xf64>
          %34 = arith.addf %31, %33 : f64
          %35 = affine.load %arg4[symbol(%0), 1] : memref<?x4xf64>
          %36 = memref.load %arg4[%32, %c1] : memref<?x4xf64>
          %37 = arith.mulf %35, %36 : f64
          %38 = affine.load %arg4[symbol(%0), 2] : memref<?x4xf64>
          %39 = memref.load %arg4[%32, %c2] : memref<?x4xf64>
          %40 = arith.mulf %38, %39 : f64
          %41 = arith.addf %37, %40 : f64
          %42 = affine.load %arg4[symbol(%0), 3] : memref<?x4xf64>
          %43 = memref.load %arg4[%32, %c3] : memref<?x4xf64>
          %44 = arith.mulf %42, %43 : f64
          %45 = arith.addf %41, %44 : f64
          %46 = arith.subf %34, %45 : f64
          %47 = arith.cmpf olt, %46, %cst : f64
          %48 = arith.select %47, %cst, %46 : f64
          %49 = func.call @pair_interaction(%arg0, %30, %arg2, %arg4, %arg5, %arg6) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
          scf.yield %30, %48 : i32, f64
        } else {
          scf.yield %arg13, %arg12 : i32, f64
        }
        scf.yield %29#1, %29#0 : f64, i32
      }
      %25 = arith.addi %arg10, %c1_i32 : i32
      scf.yield %24#0, %24#1, %25 : f64, i32, i32
    }
    return
  }
  func.func @approx_neighbor_box_accumulate_2(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg4: memref<?x4xf64>, %arg5: memref<?xf64>, %arg6: memref<?x4xf64>, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c656 = arith.constant 656 : index
    %c24 = arith.constant 24 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f64
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.muli %2, %c656 : index
    %4 = arith.index_cast %3 : index to i64
    %5 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
    %6 = llvm.getelementptr %5[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %7 = llvm.getelementptr %6[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %8 = llvm.getelementptr %6[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %9:2 = scf.while (%arg8 = %1, %arg9 = %c0_i32) : (f64, i32) -> (f64, i32) {
      %10 = llvm.load %7 : !llvm.ptr -> i32
      %11 = arith.cmpi slt, %arg9, %10 : i32
      scf.condition(%11) %arg8, %arg9 : f64, i32
    } do {
    ^bb0(%arg8: f64, %arg9: i32):
      %10 = arith.index_cast %arg9 : i32 to index
      %11 = arith.muli %10, %c24 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = llvm.getelementptr %8[%12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %14 = llvm.getelementptr %13[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
      %15 = llvm.load %14 : !llvm.ptr -> i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.muli %16, %c656 : index
      %18 = arith.index_cast %17 : index to i64
      %19 = llvm.getelementptr %5[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %20 = llvm.getelementptr %19[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %21 = llvm.load %20 : !llvm.ptr -> i64
      %22 = arith.trunci %21 : i64 to i32
      %23 = scf.for %arg10 = %c0 to %c128 step %c1 iter_args(%arg11 = %arg8) -> (f64) {
        %25 = arith.index_cast %arg10 : index to i32
        %26 = arith.addi %22, %25 : i32
        %27 = arith.andi %25, %c3_i32 : i32
        %28 = arith.cmpi ne, %27, %c3_i32 : i32
        %29 = scf.if %28 -> (f64) {
          %30 = affine.load %arg4[symbol(%0), 0] : memref<?x4xf64>
          %31 = arith.index_cast %26 : i32 to index
          %32 = memref.load %arg4[%31, %c0] : memref<?x4xf64>
          %33 = arith.addf %30, %32 : f64
          %34 = affine.load %arg4[symbol(%0), 1] : memref<?x4xf64>
          %35 = memref.load %arg4[%31, %c1] : memref<?x4xf64>
          %36 = arith.mulf %34, %35 : f64
          %37 = affine.load %arg4[symbol(%0), 2] : memref<?x4xf64>
          %38 = memref.load %arg4[%31, %c2] : memref<?x4xf64>
          %39 = arith.mulf %37, %38 : f64
          %40 = arith.addf %36, %39 : f64
          %41 = affine.load %arg4[symbol(%0), 3] : memref<?x4xf64>
          %42 = memref.load %arg4[%31, %c3] : memref<?x4xf64>
          %43 = arith.mulf %41, %42 : f64
          %44 = arith.addf %40, %43 : f64
          %45 = arith.subf %33, %44 : f64
          %46 = arith.cmpf olt, %45, %cst : f64
          %47 = arith.select %46, %cst, %45 : f64
          %48 = func.call @pair_interaction(%arg0, %26, %arg2, %arg4, %arg5, %arg6) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
          scf.yield %47 : f64
        } else {
          scf.yield %arg11 : f64
        }
        scf.yield %29 : f64
      }
      %24 = arith.addi %arg9, %c1_i32 : i32
      scf.yield %23, %24 : f64, i32
    }
    return
  }
  func.func @approx_neighbor_box_accumulate_3(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg4: memref<?x4xf64>, %arg5: memref<?xf64>, %arg6: memref<?x4xf64>, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c656 = arith.constant 656 : index
    %c24 = arith.constant 24 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f64
    %2 = llvm.mlir.undef : i32
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.muli %3, %c656 : index
    %5 = arith.index_cast %4 : index to i64
    %6 = "polygeist.memref2pointer"(%arg3) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
    %7 = llvm.getelementptr %6[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %8 = llvm.getelementptr %7[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %9 = llvm.getelementptr %7[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %10:3 = scf.while (%arg8 = %1, %arg9 = %2, %arg10 = %c0_i32) : (f64, i32, i32) -> (f64, i32, i32) {
      %11 = llvm.load %8 : !llvm.ptr -> i32
      %12 = arith.cmpi slt, %arg10, %11 : i32
      scf.condition(%12) %arg8, %arg9, %arg10 : f64, i32, i32
    } do {
    ^bb0(%arg8: f64, %arg9: i32, %arg10: i32):
      %11 = arith.index_cast %arg10 : i32 to index
      %12 = arith.muli %11, %c24 : index
      %13 = arith.index_cast %12 : index to i64
      %14 = llvm.getelementptr %9[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %15 = llvm.getelementptr %14[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
      %16 = llvm.load %15 : !llvm.ptr -> i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.muli %17, %c656 : index
      %19 = arith.index_cast %18 : index to i64
      %20 = llvm.getelementptr %6[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %21 = llvm.getelementptr %20[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %22 = llvm.load %21 : !llvm.ptr -> i64
      %23 = arith.trunci %22 : i64 to i32
      %24:2 = scf.for %arg11 = %c0 to %c128 step %c1 iter_args(%arg12 = %arg8, %arg13 = %arg9) -> (f64, i32) {
        %26 = arith.index_cast %arg11 : index to i32
        %27 = arith.andi %26, %c1_i32 : i32
        %28 = arith.cmpi eq, %27, %c0_i32 : i32
        %29:2 = scf.if %28 -> (i32, f64) {
          %30 = arith.addi %23, %26 : i32
          %31 = affine.load %arg4[symbol(%0), 0] : memref<?x4xf64>
          %32 = arith.index_cast %30 : i32 to index
          %33 = memref.load %arg4[%32, %c0] : memref<?x4xf64>
          %34 = arith.addf %31, %33 : f64
          %35 = affine.load %arg4[symbol(%0), 1] : memref<?x4xf64>
          %36 = memref.load %arg4[%32, %c1] : memref<?x4xf64>
          %37 = arith.mulf %35, %36 : f64
          %38 = affine.load %arg4[symbol(%0), 2] : memref<?x4xf64>
          %39 = memref.load %arg4[%32, %c2] : memref<?x4xf64>
          %40 = arith.mulf %38, %39 : f64
          %41 = arith.addf %37, %40 : f64
          %42 = affine.load %arg4[symbol(%0), 3] : memref<?x4xf64>
          %43 = memref.load %arg4[%32, %c3] : memref<?x4xf64>
          %44 = arith.mulf %42, %43 : f64
          %45 = arith.addf %41, %44 : f64
          %46 = arith.subf %34, %45 : f64
          %47 = arith.cmpf olt, %46, %cst : f64
          %48 = arith.select %47, %cst, %46 : f64
          %49 = func.call @pair_interaction(%arg0, %30, %arg2, %arg4, %arg5, %arg6) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
          scf.yield %30, %48 : i32, f64
        } else {
          scf.yield %arg13, %arg12 : i32, f64
        }
        scf.yield %29#1, %29#0 : f64, i32
      }
      %25 = arith.addi %arg10, %c1_i32 : i32
      scf.yield %24#0, %24#1, %25 : f64, i32, i32
    }
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %true = arith.constant true
    %false = arith.constant false
    %c656_i64 = arith.constant 656 : i64
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c656 = arith.constant 656 : index
    %c24 = arith.constant 24 : index
    %c8_i64 = arith.constant 8 : i64
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0x42008E8D71C00000 : f64
    %cst_0 = arith.constant 0xC2008E8D71C00000 : f64
    %cst_1 = arith.constant 1.000000e+06 : f64
    %cst_2 = arith.constant 1.000000e+03 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %cst_4 = arith.constant 0.000000e+00 : f64
    %cst_5 = arith.constant 1.000000e+01 : f64
    %c10_i32 = arith.constant 10 : i32
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst_6 = arith.constant 5.000000e-01 : f64
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1x2xi64>
    %alloca_7 = memref.alloca() : memref<1x2xi64>
    %alloca_8 = memref.alloca() : memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>
    %alloca_9 = memref.alloca() : memref<f64>
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr) -> memref<?x4xf64>
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = "polygeist.pointer2memref"(%5) : (!llvm.ptr) -> memref<?xf64>
    %7 = llvm.mlir.zero : !llvm.ptr
    %8 = "polygeist.pointer2memref"(%7) : (!llvm.ptr) -> memref<?x4xf64>
    %9 = llvm.mlir.addressof @str0 : !llvm.ptr
    %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
    %11 = llvm.call @printf(%10, %c128_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    %12 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
    %13 = llvm.getelementptr %12[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
    llvm.store %c1_i32, %13 : i32, !llvm.ptr
    %14 = arith.cmpi eq, %arg0, %c3_i32 : i32
    %15:3 = scf.if %14 -> (i32, i1, i32) {
      %18 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
      %19 = llvm.mlir.addressof @str1 : !llvm.ptr
      %20 = "polygeist.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi8>
      %21 = func.call @strcmp(%18, %20) : (memref<?xi8>, memref<?xi8>) -> i32
      %22 = arith.cmpi eq, %21, %c0_i32 : i32
      %23 = scf.if %22 -> (i1) {
        %25 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %26 = func.call @isInteger(%25) : (memref<?xi8>) -> i32
        %27 = arith.cmpi ne, %26, %c0_i32 : i32
        scf.yield %27 : i1
      } else {
        scf.yield %false : i1
      }
      %24:2 = scf.if %23 -> (i1, i32) {
        %25 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %26 = llvm.getelementptr %25[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %27 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
        %28 = func.call @atoi(%27) : (memref<?xi8>) -> i32
        llvm.store %28, %26 : i32, !llvm.ptr
        %29 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %30 = llvm.getelementptr %29[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %31 = llvm.load %30 : !llvm.ptr -> i32
        %32 = arith.cmpi sle, %31, %c0_i32 : i32
        %33 = arith.cmpi sgt, %31, %c0_i32 : i32
        %34 = arith.select %32, %c1_i32, %0 : i32
        scf.if %32 {
          %35 = llvm.mlir.addressof @str2 : !llvm.ptr
          %36 = llvm.getelementptr %35[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
          %37 = llvm.call @printf(%36) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        scf.yield %33, %34 : i1, i32
      } else {
        %25 = llvm.mlir.addressof @str3 : !llvm.ptr
        %26 = llvm.getelementptr %25[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
        %27 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
        %28 = "polygeist.memref2pointer"(%27) : (memref<?xi8>) -> !llvm.ptr
        %29 = llvm.call @printf(%26, %28) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        scf.yield %false, %c1_i32 : i1, i32
      }
      scf.yield %c2_i32, %24#0, %24#1 : i32, i1, i32
    } else {
      %18 = arith.cmpi eq, %arg0, %c4_i32 : i32
      %19:3 = scf.if %18 -> (i32, i1, i32) {
        %20 = affine.load %arg1[1] : memref<?xmemref<?xi8>>
        %21 = llvm.mlir.addressof @str1 : !llvm.ptr
        %22 = "polygeist.pointer2memref"(%21) : (!llvm.ptr) -> memref<?xi8>
        %23 = func.call @strcmp(%20, %22) : (memref<?xi8>, memref<?xi8>) -> i32
        %24 = arith.cmpi eq, %23, %c0_i32 : i32
        %25 = scf.if %24 -> (i1) {
          %27 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
          %28 = func.call @isInteger(%27) : (memref<?xi8>) -> i32
          %29 = arith.cmpi ne, %28, %c0_i32 : i32
          scf.yield %29 : i1
        } else {
          scf.yield %false : i1
        }
        %26:3 = scf.if %25 -> (i32, i1, i32) {
          %27 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %28 = llvm.getelementptr %27[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %29 = affine.load %arg1[2] : memref<?xmemref<?xi8>>
          %30 = func.call @atoi(%29) : (memref<?xi8>) -> i32
          llvm.store %30, %28 : i32, !llvm.ptr
          %31 = affine.load %arg1[3] : memref<?xmemref<?xi8>>
          %32 = func.call @atoi(%31) : (memref<?xi8>) -> i32
          %33 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
          %34 = llvm.getelementptr %33[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
          %35 = llvm.load %34 : !llvm.ptr -> i32
          %36 = arith.cmpi sle, %35, %c0_i32 : i32
          %37 = arith.cmpi sgt, %35, %c0_i32 : i32
          %38 = arith.select %36, %c1_i32, %0 : i32
          scf.if %36 {
            %39 = llvm.mlir.addressof @str2 : !llvm.ptr
            %40 = llvm.getelementptr %39[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
            %41 = llvm.call @printf(%40) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
          }
          scf.yield %32, %37, %38 : i32, i1, i32
        } else {
          %27 = llvm.mlir.addressof @str3 : !llvm.ptr
          %28 = llvm.getelementptr %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<36 x i8>
          %29 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
          %30 = "polygeist.memref2pointer"(%29) : (memref<?xi8>) -> !llvm.ptr
          %31 = llvm.call @printf(%28, %30) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
          scf.yield %c2_i32, %false, %c1_i32 : i32, i1, i32
        }
        scf.yield %26#0, %26#1, %26#2 : i32, i1, i32
      } else {
        %20 = llvm.mlir.addressof @str4 : !llvm.ptr
        %21 = llvm.getelementptr %20[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<29 x i8>
        %22 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
        %23 = "polygeist.memref2pointer"(%22) : (memref<?xi8>) -> !llvm.ptr
        %24 = llvm.call @printf(%21, %23) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
        scf.yield %c2_i32, %false, %c1_i32 : i32, i1, i32
      }
      scf.yield %19#0, %19#1, %19#2 : i32, i1, i32
    }
    %16:6 = scf.if %15#1 -> (i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) {
      %18 = llvm.mlir.addressof @str5 : !llvm.ptr
      %19 = llvm.getelementptr %18[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<29 x i8>
      %20 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %21 = llvm.getelementptr %20[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %22 = llvm.load %21 : !llvm.ptr -> i32
      %23 = llvm.call @printf(%19, %22) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      affine.store %cst_6, %alloca_9[] : memref<f64>
      %24 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %25 = llvm.getelementptr %24[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %26 = llvm.getelementptr %24[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %27 = llvm.load %26 : !llvm.ptr -> i32
      %28 = arith.muli %27, %27 : i32
      %29 = arith.muli %28, %27 : i32
      %30 = arith.extsi %29 : i32 to i64
      llvm.store %30, %25 : i64, !llvm.ptr
      %31 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %32 = llvm.getelementptr %31[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %33 = llvm.getelementptr %31[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %34 = llvm.load %33 : !llvm.ptr -> i64
      %35 = arith.muli %34, %c128_i64 : i64
      llvm.store %35, %32 : i64, !llvm.ptr
      %36 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %37 = llvm.getelementptr %36[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %38 = llvm.load %37 : !llvm.ptr -> i64
      %39 = arith.muli %38, %c656_i64 : i64
      %40 = arith.index_cast %39 : i64 to index
      %41 = arith.divui %40, %c656 : index
      %alloc = memref.alloc(%41) : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
      %42 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %43 = llvm.getelementptr %42[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %44 = llvm.load %43 : !llvm.ptr -> i64
      %45 = arith.muli %44, %c32_i64 : i64
      %46 = arith.index_cast %45 : i64 to index
      %47 = arith.divui %46, %c32 : index
      %alloc_10 = memref.alloc(%47) : memref<?x4xf64>
      %48 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %49 = llvm.getelementptr %48[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %50 = llvm.load %49 : !llvm.ptr -> i64
      %51 = arith.muli %50, %c8_i64 : i64
      %52 = arith.index_cast %51 : i64 to index
      %53 = arith.divui %52, %c8 : index
      %alloc_11 = memref.alloc(%53) : memref<?xf64>
      %54 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %55 = llvm.getelementptr %54[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %56 = llvm.load %55 : !llvm.ptr -> i64
      %57 = arith.muli %56, %c32_i64 : i64
      %58 = arith.index_cast %57 : i64 to index
      %59 = arith.divui %58, %c32 : index
      %alloc_12 = memref.alloc(%59) : memref<?x4xf64>
      %60 = "polygeist.memref2pointer"(%alloc) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
      %61 = llvm.mlir.zero : !llvm.ptr
      %62 = llvm.icmp "eq" %60, %61 : !llvm.ptr
      %63 = scf.if %62 -> (i1) {
        scf.yield %true : i1
      } else {
        %68 = "polygeist.memref2pointer"(%alloc_10) : (memref<?x4xf64>) -> !llvm.ptr
        %69 = llvm.icmp "eq" %68, %61 : !llvm.ptr
        scf.yield %69 : i1
      }
      %64 = scf.if %63 -> (i1) {
        scf.yield %true : i1
      } else {
        %68 = "polygeist.memref2pointer"(%alloc_11) : (memref<?xf64>) -> !llvm.ptr
        %69 = llvm.icmp "eq" %68, %61 : !llvm.ptr
        scf.yield %69 : i1
      }
      %65 = scf.if %64 -> (i1) {
        scf.yield %true : i1
      } else {
        %68 = "polygeist.memref2pointer"(%alloc_12) : (memref<?x4xf64>) -> !llvm.ptr
        %69 = llvm.icmp "eq" %68, %61 : !llvm.ptr
        scf.yield %69 : i1
      }
      %66 = arith.xori %65, %true : i1
      %67 = arith.select %65, %c1_i32, %15#2 : i32
      scf.if %65 {
        %68 = llvm.mlir.addressof @str6 : !llvm.ptr
        %69 = llvm.getelementptr %68[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12 x i8>
        %70 = llvm.call @printf(%69) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        memref.dealloc %alloc_10 : memref<?x4xf64>
        memref.dealloc %alloc_11 : memref<?xf64>
        memref.dealloc %alloc_12 : memref<?x4xf64>
        memref.dealloc %alloc : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
      }
      scf.yield %66, %67, %alloc_12, %alloc_11, %alloc_10, %alloc : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
    } else {
      scf.yield %false, %15#2, %8, %6, %4, %2 : i1, i32, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
    }
    %17 = arith.select %16#0, %c0_i32, %16#1 : i32
    scf.if %16#0 {
      %18 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %19 = llvm.getelementptr %18[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %20 = "polygeist.memref2pointer"(%16#5) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
      %21:2 = scf.while (%arg2 = %c0_i32, %arg3 = %c0_i32) : (i32, i32) -> (i32, i32) {
        %133 = llvm.load %19 : !llvm.ptr -> i32
        %134 = arith.cmpi slt, %arg2, %133 : i32
        scf.condition(%134) %arg2, %arg3 : i32, i32
      } do {
      ^bb0(%arg2: i32, %arg3: i32):
        %133:2 = scf.while (%arg4 = %c0_i32, %arg5 = %arg3) : (i32, i32) -> (i32, i32) {
          %135 = llvm.load %19 : !llvm.ptr -> i32
          %136 = arith.cmpi slt, %arg4, %135 : i32
          scf.condition(%136) %arg5, %arg4 : i32, i32
        } do {
        ^bb0(%arg4: i32, %arg5: i32):
          %135:2 = scf.while (%arg6 = %c0_i32, %arg7 = %arg4) : (i32, i32) -> (i32, i32) {
            %137 = llvm.load %19 : !llvm.ptr -> i32
            %138 = arith.cmpi slt, %arg6, %137 : i32
            scf.condition(%138) %arg7, %arg6 : i32, i32
          } do {
          ^bb0(%arg6: i32, %arg7: i32):
            %137 = arith.index_cast %arg6 : i32 to index
            %138 = arith.muli %137, %c656 : index
            %139 = arith.index_cast %138 : index to i64
            %140 = llvm.getelementptr %20[%139] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %141 = llvm.getelementptr %140[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            llvm.store %arg6, %141 : i32, !llvm.ptr
            %142 = llvm.getelementptr %140[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            %143 = arith.muli %arg6, %c128_i32 : i32
            %144 = arith.extsi %143 : i32 to i64
            llvm.store %144, %142 : i64, !llvm.ptr
            %145 = llvm.getelementptr %140[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
            llvm.store %c0_i32, %145 : i32, !llvm.ptr
            scf.for %arg8 = %c-1 to %c2 step %c1 {
              %148 = arith.index_cast %arg8 : index to i32
              scf.for %arg9 = %c-1 to %c2 step %c1 {
                %149 = arith.index_cast %arg9 : index to i32
                scf.for %arg10 = %c-1 to %c2 step %c1 {
                  %150 = arith.index_cast %arg10 : index to i32
                  %151 = arith.cmpi eq, %148, %c0_i32 : i32
                  %152 = arith.cmpi eq, %149, %c0_i32 : i32
                  %153 = arith.andi %151, %152 : i1
                  %154 = scf.if %153 -> (i1) {
                    %161 = arith.cmpi eq, %150, %c0_i32 : i32
                    scf.yield %161 : i1
                  } else {
                    scf.yield %false : i1
                  }
                  %155 = arith.xori %154, %true : i1
                  %156 = arith.addi %arg2, %148 : i32
                  %157 = arith.addi %arg5, %149 : i32
                  %158 = arith.addi %arg7, %150 : i32
                  %159 = arith.cmpi sge, %156, %c0_i32 : i32
                  %160 = arith.andi %155, %159 : i1
                  scf.if %160 {
                    %161 = llvm.load %19 : !llvm.ptr -> i32
                    %162 = arith.cmpi slt, %156, %161 : i32
                    %163 = arith.cmpi sge, %157, %c0_i32 : i32
                    %164 = arith.andi %162, %163 : i1
                    scf.if %164 {
                      %165 = llvm.load %19 : !llvm.ptr -> i32
                      %166 = arith.cmpi slt, %157, %165 : i32
                      %167 = arith.cmpi sge, %158, %c0_i32 : i32
                      %168 = arith.andi %166, %167 : i1
                      scf.if %168 {
                        %169 = llvm.load %19 : !llvm.ptr -> i32
                        %170 = arith.cmpi slt, %158, %169 : i32
                        scf.if %170 {
                          %171 = llvm.load %145 : !llvm.ptr -> i32
                          %172 = arith.addi %171, %c1_i32 : i32
                          llvm.store %172, %145 : i32, !llvm.ptr
                          %173 = llvm.load %19 : !llvm.ptr -> i32
                          %174 = arith.muli %156, %173 : i32
                          %175 = arith.muli %174, %173 : i32
                          %176 = arith.muli %157, %173 : i32
                          %177 = arith.addi %175, %176 : i32
                          %178 = arith.addi %177, %158 : i32
                          %179 = llvm.getelementptr %140[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
                          %180 = arith.index_cast %171 : i32 to index
                          %181 = arith.muli %180, %c24 : index
                          %182 = arith.index_cast %181 : index to i64
                          %183 = llvm.getelementptr %179[%182] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                          %184 = llvm.getelementptr %183[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
                          llvm.store %178, %184 : i32, !llvm.ptr
                          %185 = llvm.getelementptr %183[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64)>
                          %186 = arith.muli %178, %c128_i32 : i32
                          %187 = arith.extsi %186 : i32 to i64
                          llvm.store %187, %185 : i64, !llvm.ptr
                        }
                      }
                    }
                  }
                }
              }
            }
            %146 = arith.addi %arg6, %c1_i32 : i32
            %147 = arith.addi %arg7, %c1_i32 : i32
            scf.yield %147, %146 : i32, i32
          }
          %136 = arith.addi %arg5, %c1_i32 : i32
          scf.yield %136, %135#0 : i32, i32
        }
        %134 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %134, %133#0 : i32, i32
      }
      func.call @srand(%15#0) : (i32) -> ()
      %22 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %23 = llvm.getelementptr %22[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %24 = scf.while (%arg2 = %c0_i64) : (i64) -> i64 {
        %133 = llvm.load %23 : !llvm.ptr -> i64
        %134 = arith.cmpi slt, %arg2, %133 : i64
        scf.condition(%134) %arg2 : i64
      } do {
      ^bb0(%arg2: i64):
        %133 = arith.index_cast %arg2 : i64 to index
        %134 = func.call @rand() : () -> i32
        %135 = arith.remsi %134, %c10_i32 : i32
        %136 = arith.addi %135, %c1_i32 : i32
        %137 = arith.sitofp %136 : i32 to f64
        %138 = arith.divf %137, %cst_5 : f64
        memref.store %138, %16#4[%133, %c0] : memref<?x4xf64>
        %139 = func.call @rand() : () -> i32
        %140 = arith.remsi %139, %c10_i32 : i32
        %141 = arith.addi %140, %c1_i32 : i32
        %142 = arith.sitofp %141 : i32 to f64
        %143 = arith.divf %142, %cst_5 : f64
        memref.store %143, %16#4[%133, %c1] : memref<?x4xf64>
        %144 = func.call @rand() : () -> i32
        %145 = arith.remsi %144, %c10_i32 : i32
        %146 = arith.addi %145, %c1_i32 : i32
        %147 = arith.sitofp %146 : i32 to f64
        %148 = arith.divf %147, %cst_5 : f64
        memref.store %148, %16#4[%133, %c2] : memref<?x4xf64>
        %149 = func.call @rand() : () -> i32
        %150 = arith.remsi %149, %c10_i32 : i32
        %151 = arith.addi %150, %c1_i32 : i32
        %152 = arith.sitofp %151 : i32 to f64
        %153 = arith.divf %152, %cst_5 : f64
        memref.store %153, %16#4[%133, %c3] : memref<?x4xf64>
        %154 = func.call @rand() : () -> i32
        %155 = arith.remsi %154, %c10_i32 : i32
        %156 = arith.addi %155, %c1_i32 : i32
        %157 = arith.sitofp %156 : i32 to f64
        %158 = arith.divf %157, %cst_5 : f64
        memref.store %158, %16#3[%133] : memref<?xf64>
        memref.store %cst_4, %16#2[%133, %c3] : memref<?x4xf64>
        %159 = memref.load %16#2[%133, %c3] : memref<?x4xf64>
        memref.store %159, %16#2[%133, %c2] : memref<?x4xf64>
        %160 = memref.load %16#2[%133, %c2] : memref<?x4xf64>
        memref.store %160, %16#2[%133, %c1] : memref<?x4xf64>
        %161 = memref.load %16#2[%133, %c1] : memref<?x4xf64>
        memref.store %161, %16#2[%133, %c0] : memref<?x4xf64>
        %162 = arith.addi %arg2, %c1_i64 : i64
        scf.yield %162 : i64
      }
      %cast = memref.cast %alloca_7 : memref<1x2xi64> to memref<?x2xi64>
      %25 = func.call @clock_gettime(%c1_i32, %cast) : (i32, memref<?x2xi64>) -> i32
      %26 = affine.load %alloca_9[] : memref<f64>
      %27 = arith.mulf %26, %cst_3 : f64
      %28 = arith.mulf %27, %26 : f64
      %29 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %30 = llvm.getelementptr %29[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %31 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
        %133 = arith.extsi %arg2 : i32 to i64
        %134 = llvm.load %30 : !llvm.ptr -> i64
        %135 = arith.cmpi slt, %133, %134 : i64
        scf.condition(%135) %arg2 : i32
      } do {
      ^bb0(%arg2: i32):
        func.call @process_home_box(%arg2, %28, %16#5, %16#4, %16#3, %16#2) : (i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> ()
        %133 = arith.addi %arg2, %c1_i32 : i32
        scf.yield %133 : i32
      }
      %cast_10 = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
      %32 = func.call @clock_gettime(%c1_i32, %cast_10) : (i32, memref<?x2xi64>) -> i32
      %33 = affine.load %alloca[0, 0] : memref<1x2xi64>
      %34 = affine.load %alloca_7[0, 0] : memref<1x2xi64>
      %35 = arith.subi %33, %34 : i64
      %36 = arith.sitofp %35 : i64 to f64
      %37 = arith.mulf %36, %cst_2 : f64
      %38 = affine.load %alloca[0, 1] : memref<1x2xi64>
      %39 = affine.load %alloca_7[0, 1] : memref<1x2xi64>
      %40 = arith.subi %38, %39 : i64
      %41 = arith.sitofp %40 : i64 to f64
      %42 = arith.divf %41, %cst_1 : f64
      %43 = arith.addf %37, %42 : f64
      %44 = llvm.mlir.addressof @str7 : !llvm.ptr
      %45 = llvm.getelementptr %44[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31 x i8>
      %46 = llvm.call @printf(%45, %43) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      %47 = llvm.mlir.addressof @str8 : !llvm.ptr
      %48 = llvm.getelementptr %47[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
      %49 = llvm.call @printf(%48) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %50 = llvm.mlir.addressof @str9 : !llvm.ptr
      %51 = llvm.getelementptr %50[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
      %52 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %53 = llvm.getelementptr %52[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %54 = llvm.load %53 : !llvm.ptr -> i64
      %55 = llvm.call @printf(%51, %54) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      %56 = llvm.mlir.addressof @str10 : !llvm.ptr
      %57 = llvm.getelementptr %56[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
      %58 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %59 = llvm.getelementptr %58[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %60 = llvm.load %59 : !llvm.ptr -> i64
      %61 = llvm.call @printf(%57, %60) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
      %62 = llvm.mlir.addressof @str11 : !llvm.ptr
      %63 = llvm.getelementptr %62[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
      %64 = llvm.call @printf(%63, %c128_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %65 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %66 = llvm.getelementptr %65[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %67 = llvm.load %66 : !llvm.ptr -> i64
      %68 = arith.index_cast %67 : i64 to index
      %69:12 = scf.for %arg2 = %c0 to %68 step %c1 iter_args(%arg3 = %cst, %arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_4, %arg12 = %cst_4, %arg13 = %cst_4, %arg14 = %cst_4) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        %133 = memref.load %16#2[%arg2, %c0] : memref<?x4xf64>
        %134 = arith.addf %arg14, %133 : f64
        %135 = memref.load %16#2[%arg2, %c1] : memref<?x4xf64>
        %136 = arith.addf %arg13, %135 : f64
        %137 = memref.load %16#2[%arg2, %c2] : memref<?x4xf64>
        %138 = arith.addf %arg12, %137 : f64
        %139 = memref.load %16#2[%arg2, %c3] : memref<?x4xf64>
        %140 = arith.addf %arg11, %139 : f64
        %141 = arith.cmpf ogt, %133, %arg10 : f64
        %142 = scf.if %141 -> (f64) {
          %157 = memref.load %16#2[%arg2, %c0] : memref<?x4xf64>
          scf.yield %157 : f64
        } else {
          scf.yield %arg10 : f64
        }
        %143 = arith.cmpf ogt, %135, %arg9 : f64
        %144 = scf.if %143 -> (f64) {
          %157 = memref.load %16#2[%arg2, %c1] : memref<?x4xf64>
          scf.yield %157 : f64
        } else {
          scf.yield %arg9 : f64
        }
        %145 = arith.cmpf ogt, %137, %arg8 : f64
        %146 = scf.if %145 -> (f64) {
          %157 = memref.load %16#2[%arg2, %c2] : memref<?x4xf64>
          scf.yield %157 : f64
        } else {
          scf.yield %arg8 : f64
        }
        %147 = arith.cmpf ogt, %139, %arg7 : f64
        %148 = scf.if %147 -> (f64) {
          %157 = memref.load %16#2[%arg2, %c3] : memref<?x4xf64>
          scf.yield %157 : f64
        } else {
          scf.yield %arg7 : f64
        }
        %149 = arith.cmpf olt, %133, %arg6 : f64
        %150 = scf.if %149 -> (f64) {
          %157 = memref.load %16#2[%arg2, %c0] : memref<?x4xf64>
          scf.yield %157 : f64
        } else {
          scf.yield %arg6 : f64
        }
        %151 = arith.cmpf olt, %135, %arg5 : f64
        %152 = scf.if %151 -> (f64) {
          %157 = memref.load %16#2[%arg2, %c1] : memref<?x4xf64>
          scf.yield %157 : f64
        } else {
          scf.yield %arg5 : f64
        }
        %153 = arith.cmpf olt, %137, %arg4 : f64
        %154 = scf.if %153 -> (f64) {
          %157 = memref.load %16#2[%arg2, %c2] : memref<?x4xf64>
          scf.yield %157 : f64
        } else {
          scf.yield %arg4 : f64
        }
        %155 = arith.cmpf olt, %139, %arg3 : f64
        %156 = scf.if %155 -> (f64) {
          %157 = memref.load %16#2[%arg2, %c3] : memref<?x4xf64>
          scf.yield %157 : f64
        } else {
          scf.yield %arg3 : f64
        }
        scf.yield %156, %154, %152, %150, %148, %146, %144, %142, %140, %138, %136, %134 : f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
      }
      %70 = llvm.mlir.addressof @str12 : !llvm.ptr
      %71 = llvm.getelementptr %70[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x i8>
      %72 = llvm.call @printf(%71) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %73 = llvm.mlir.addressof @str13 : !llvm.ptr
      %74 = llvm.getelementptr %73[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x i8>
      %75 = llvm.call @printf(%74) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %76 = llvm.mlir.addressof @str14 : !llvm.ptr
      %77 = llvm.getelementptr %76[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<65 x i8>
      %78 = llvm.call @printf(%77) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %79 = llvm.mlir.addressof @str15 : !llvm.ptr
      %80 = llvm.getelementptr %79[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<49 x i8>
      %81 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %82 = llvm.getelementptr %81[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %83 = llvm.load %82 : !llvm.ptr -> i64
      %84 = arith.sitofp %83 : i64 to f64
      %85 = arith.divf %69#11, %84 : f64
      %86 = llvm.call @printf(%80, %85, %69#3, %69#7) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %87 = llvm.mlir.addressof @str16 : !llvm.ptr
      %88 = llvm.getelementptr %87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %89 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %90 = llvm.getelementptr %89[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %91 = llvm.load %90 : !llvm.ptr -> i64
      %92 = arith.sitofp %91 : i64 to f64
      %93 = arith.divf %69#10, %92 : f64
      %94 = llvm.call @printf(%88, %93, %69#2, %69#6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %95 = llvm.mlir.addressof @str17 : !llvm.ptr
      %96 = llvm.getelementptr %95[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %97 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %98 = llvm.getelementptr %97[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %99 = llvm.load %98 : !llvm.ptr -> i64
      %100 = arith.sitofp %99 : i64 to f64
      %101 = arith.divf %69#9, %100 : f64
      %102 = llvm.call @printf(%96, %101, %69#1, %69#5) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %103 = llvm.mlir.addressof @str18 : !llvm.ptr
      %104 = llvm.getelementptr %103[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<51 x i8>
      %105 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %106 = llvm.getelementptr %105[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %107 = llvm.load %106 : !llvm.ptr -> i64
      %108 = arith.sitofp %107 : i64 to f64
      %109 = arith.divf %69#8, %108 : f64
      %110 = llvm.call @printf(%104, %109, %69#0, %69#4) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64, f64) -> i32
      %111 = llvm.mlir.addressof @str19 : !llvm.ptr
      %112 = llvm.getelementptr %111[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
      %113 = llvm.call @printf(%112) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %114 = llvm.mlir.addressof @str20 : !llvm.ptr
      %115 = llvm.getelementptr %114[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<74 x i8>
      %116 = llvm.call @printf(%115) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %117 = llvm.mlir.addressof @str21 : !llvm.ptr
      %118 = llvm.getelementptr %117[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<75 x i8>
      %119 = llvm.call @printf(%118) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %120 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
      %121 = llvm.getelementptr %120[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
      %122 = llvm.mlir.addressof @str22 : !llvm.ptr
      %123 = llvm.getelementptr %122[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<43 x i8>
      %124 = scf.while (%arg2 = %c0_i64) : (i64) -> i64 {
        %133 = llvm.load %121 : !llvm.ptr -> i64
        %134 = arith.cmpi slt, %arg2, %133 : i64
        scf.condition(%134) %arg2 : i64
      } do {
      ^bb0(%arg2: i64):
        %133 = arith.index_cast %arg2 : i64 to index
        %134 = memref.load %16#2[%133, %c0] : memref<?x4xf64>
        %135 = memref.load %16#2[%133, %c1] : memref<?x4xf64>
        %136 = memref.load %16#2[%133, %c2] : memref<?x4xf64>
        %137 = memref.load %16#2[%133, %c3] : memref<?x4xf64>
        %138 = llvm.call @printf(%123, %arg2, %134, %135, %136, %137) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, f64, f64, f64, f64) -> i32
        %139 = arith.addi %arg2, %c1_i64 : i64
        scf.yield %139 : i64
      }
      %125 = llvm.mlir.addressof @str23 : !llvm.ptr
      %126 = llvm.mlir.addressof @str24 : !llvm.ptr
      %127 = "polygeist.pointer2memref"(%125) : (!llvm.ptr) -> memref<?xi8>
      %128 = "polygeist.pointer2memref"(%126) : (!llvm.ptr) -> memref<?xi8>
      %129 = func.call @fopen(%127, %128) : (memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>
      %130 = "polygeist.memref2pointer"(%129) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> !llvm.ptr
      %131 = llvm.mlir.zero : !llvm.ptr
      %132 = llvm.icmp "ne" %130, %131 : !llvm.ptr
      scf.if %132 {
        %133 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %134 = llvm.getelementptr %133[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %135 = "polygeist.pointer2memref"(%134) : (!llvm.ptr) -> memref<?xi8>
        %136 = func.call @fwrite(%135, %c8_i64, %c1_i64, %129) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %137 = arith.cmpi ne, %136, %c1_i64 : i64
        scf.if %137 {
          %152 = llvm.mlir.addressof @str25 : !llvm.ptr
          %153 = llvm.getelementptr %152[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
          %154 = llvm.call @printf(%153) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %138 = "polygeist.memref2pointer"(%16#2) : (memref<?x4xf64>) -> !llvm.ptr
        %139 = "polygeist.pointer2memref"(%138) : (!llvm.ptr) -> memref<?xi8>
        %140 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %141 = llvm.getelementptr %140[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %142 = llvm.load %141 : !llvm.ptr -> i64
        %143 = func.call @fwrite(%139, %c32_i64, %142, %129) : (memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64
        %144 = "polygeist.memref2pointer"(%alloca_8) : (memref<1x!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>>) -> !llvm.ptr
        %145 = llvm.getelementptr %144[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
        %146 = llvm.load %145 : !llvm.ptr -> i64
        %147 = arith.cmpi ne, %143, %146 : i64
        scf.if %147 {
          %152 = llvm.mlir.addressof @str26 : !llvm.ptr
          %153 = llvm.getelementptr %152[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<19 x i8>
          %154 = llvm.call @printf(%153) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
        %148 = func.call @fclose(%129) : (memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32
        %149 = llvm.mlir.addressof @str27 : !llvm.ptr
        %150 = llvm.getelementptr %149[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31 x i8>
        %151 = llvm.call @printf(%150) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      } else {
        %133 = llvm.mlir.addressof @str28 : !llvm.ptr
        %134 = llvm.getelementptr %133[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
        %135 = llvm.call @printf(%134) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      }
      memref.dealloc %16#4 : memref<?x4xf64>
      memref.dealloc %16#3 : memref<?xf64>
      memref.dealloc %16#2 : memref<?x4xf64>
      memref.dealloc %16#5 : memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>
    }
    return %17 : i32
  }
  func.func private @strcmp(memref<?xi8>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @isInteger(%arg0: memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c57_i32 = arith.constant 57 : i32
    %c48_i32 = arith.constant 48 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c1 = arith.constant 1 : index
    %0 = llvm.mlir.undef : i32
    %1 = "polygeist.memref2pointer"(%arg0) : (memref<?xi8>) -> !llvm.ptr
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "eq" %1, %2 : !llvm.ptr
    %4 = scf.if %3 -> (i1) {
      scf.yield %true : i1
    } else {
      %8 = affine.load %arg0[0] : memref<?xi8>
      %9 = arith.cmpi eq, %8, %c0_i8 : i8
      scf.yield %9 : i1
    }
    %5 = arith.select %4, %c0_i32, %0 : i32
    %6:2 = scf.if %4 -> (i1, i32) {
      scf.yield %false, %5 : i1, i32
    } else {
      %8:4 = scf.while (%arg1 = %arg0, %arg2 = %true, %arg3 = %5, %arg4 = %true) : (memref<?xi8>, i1, i32, i1) -> (i1, i32, i8, memref<?xi8>) {
        %9 = affine.load %arg1[0] : memref<?xi8>
        %10 = arith.cmpi ne, %9, %c0_i8 : i8
        %11 = arith.andi %10, %arg4 : i1
        scf.condition(%11) %arg2, %arg3, %9, %arg1 : i1, i32, i8, memref<?xi8>
      } do {
      ^bb0(%arg1: i1, %arg2: i32, %arg3: i8, %arg4: memref<?xi8>):
        %9 = arith.extui %arg3 : i8 to i32
        %10 = arith.cmpi slt, %9, %c48_i32 : i32
        %11 = scf.if %10 -> (i1) {
          scf.yield %true : i1
        } else {
          %17 = arith.extui %arg3 : i8 to i32
          %18 = arith.cmpi sgt, %17, %c57_i32 : i32
          scf.yield %18 : i1
        }
        %12 = arith.xori %11, %true : i1
        %13 = arith.andi %12, %arg1 : i1
        %14 = arith.select %11, %c0_i32, %arg2 : i32
        %15 = arith.xori %11, %true : i1
        %16 = scf.if %11 -> (memref<?xi8>) {
          scf.yield %arg4 : memref<?xi8>
        } else {
          %17 = "polygeist.subindex"(%arg4, %c1) : (memref<?xi8>, index) -> memref<?xi8>
          scf.yield %17 : memref<?xi8>
        }
        scf.yield %16, %13, %14, %15 : memref<?xi8>, i1, i32, i1
      }
      scf.yield %8#0, %8#1 : i1, i32
    }
    %7 = arith.select %6#0, %c1_i32, %6#1 : i32
    return %7 : i32
  }
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @srand(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @clock_gettime(i32, memref<?x2xi64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @process_home_box(%arg0: i32, %arg1: f64, %arg2: memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c656 = arith.constant 656 : index
    %cst = arith.constant 1.000000e+02 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %alloca = memref.alloca() : memref<1x4xf64>
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.muli %0, %c656 : index
    %2 = arith.index_cast %1 : index to i64
    %3 = "polygeist.memref2pointer"(%arg2) : (memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>) -> !llvm.ptr
    %4 = llvm.getelementptr %3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %5 = llvm.getelementptr %4[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
    %6 = llvm.load %5 : !llvm.ptr -> i64
    %7 = arith.trunci %6 : i64 to i32
    scf.for %arg6 = %c0 to %c128 step %c1 {
      %8 = arith.index_cast %arg6 : index to i32
      %9 = arith.addi %7, %8 : i32
      affine.store %cst_0, %alloca[0, 0] : memref<1x4xf64>
      affine.store %cst_0, %alloca[0, 1] : memref<1x4xf64>
      affine.store %cst_0, %alloca[0, 2] : memref<1x4xf64>
      affine.store %cst_0, %alloca[0, 3] : memref<1x4xf64>
      %10 = arith.index_cast %9 : i32 to index
      %11 = memref.load %arg4[%10] : memref<?xf64>
      %12 = math.absf %11 : f64
      %13 = arith.mulf %12, %cst : f64
      %14 = arith.fptosi %13 : f64 to i32
      %15 = llvm.getelementptr %4[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>
      %16 = llvm.load %15 : !llvm.ptr -> i32
      %cast = memref.cast %alloca : memref<1x4xf64> to memref<?x4xf64>
      func.call @self_box_accumulate(%9, %7, %arg1, %arg3, %arg4, %cast, %14) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> ()
      func.call @neighbor_box_accumulate(%9, %arg0, %arg1, %arg2, %arg3, %arg4, %cast, %16) : (i32, i32, f64, memref<?x!llvm.struct<(i32, i32, i32, i32, i64, i32, array<26 x struct<(i32, i32, i32, i32, i64)>>)>>, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>, i32) -> ()
      %17 = affine.load %alloca[0, 0] : memref<1x4xf64>
      %18 = memref.load %arg5[%10, %c0] : memref<?x4xf64>
      %19 = arith.addf %18, %17 : f64
      memref.store %19, %arg5[%10, %c0] : memref<?x4xf64>
      %20 = affine.load %alloca[0, 1] : memref<1x4xf64>
      %21 = memref.load %arg5[%10, %c1] : memref<?x4xf64>
      %22 = arith.addf %21, %20 : f64
      memref.store %22, %arg5[%10, %c1] : memref<?x4xf64>
      %23 = affine.load %alloca[0, 2] : memref<1x4xf64>
      %24 = memref.load %arg5[%10, %c2] : memref<?x4xf64>
      %25 = arith.addf %24, %23 : f64
      memref.store %25, %arg5[%10, %c2] : memref<?x4xf64>
      %26 = affine.load %alloca[0, 3] : memref<1x4xf64>
      %27 = memref.load %arg5[%10, %c3] : memref<?x4xf64>
      %28 = arith.addf %27, %26 : f64
      memref.store %28, %arg5[%10, %c3] : memref<?x4xf64>
    }
    return
  }
  func.func private @fopen(memref<?xi8>, memref<?xi8>) -> memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>> attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fwrite(memref<?xi8>, i64, i64, memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @fclose(memref<?x!llvm.struct<(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, array<1 x i8>, ptr, i64, ptr, ptr, ptr, ptr, i64, i32, array<20 x i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @self_box_accumulate(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x4xf64>, %arg4: memref<?xf64>, %arg5: memref<?x4xf64>, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c60_i32 = arith.constant 60 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %0 = call @approx_state_identity(%arg6) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c60_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.index_switch %3 
    case 0 {
      scf.for %arg7 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg7 : index to i32
        %5 = arith.addi %arg1, %4 : i32
        %6 = arith.cmpi ne, %arg0, %5 : i32
        scf.if %6 {
          %7 = func.call @pair_interaction(%arg0, %5, %arg2, %arg3, %arg4, %arg5) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
        }
      }
      scf.yield
    }
    case 1 {
      scf.for %arg7 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg7 : index to i32
        %5 = arith.addi %arg1, %4 : i32
        %6 = arith.cmpi ne, %arg0, %5 : i32
        scf.if %6 {
          %7 = func.call @pair_interaction(%arg0, %5, %arg2, %arg3, %arg4, %arg5) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
        }
      }
      scf.yield
    }
    default {
      scf.for %arg7 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg7 : index to i32
        %5 = arith.addi %arg1, %4 : i32
        %6 = arith.cmpi ne, %arg0, %5 : i32
        scf.if %6 {
          %7 = func.call @pair_interaction(%arg0, %5, %arg2, %arg3, %arg4, %arg5) : (i32, i32, f64, memref<?x4xf64>, memref<?xf64>, memref<?x4xf64>) -> i32
        }
      }
    }
    return
  }
}

