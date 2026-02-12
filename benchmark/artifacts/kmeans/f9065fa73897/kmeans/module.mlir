module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<[8, 32]> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<[16, 32]> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu", "polygeist.target-cpu" = "generic", "polygeist.target-features" = "+fp-armv8,+neon,+outline-atomics,+v8a,-fmv"} {
  llvm.mlir.global internal constant @str22("\0AK-means completed in %.3f ms\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str21("  Max iterations: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str20("  Clusters: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str19("  Dimensions: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str18("  Points: %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str17("Running K-means with:\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str16("Usage: %s [-n num_points] [-d dimensions] [-k clusters] [-i max_iterations]\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str15("-h\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str14("-s\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str13("-i\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str12("-k\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str11("-d\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str10("-n\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str9("  Point %d: (\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str8("Cluster %d samples:\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str7("\0ASample points from each cluster:\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str6("Cluster %d: %d points\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str5("\0ACluster sizes:\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str4(")\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str3(", \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("%.2f\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("Centroid %d: (\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str0("Final centroids:\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  memref.global @seed : memref<1xi32> = dense<42>
  func.func @approx_state_identity(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    return %arg0 : i32
  }
  func.func @compute_distance_sq(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: i32) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = scf.for %arg4 = %c0 to %0 step %c1 iter_args(%arg5 = %cst) -> (f64) {
      %2 = memref.load %arg0[%arg4] : memref<?xf64>
      %3 = memref.load %arg1[%arg4] : memref<?xf64>
      %4 = arith.subf %2, %3 : f64
      %5 = arith.mulf %4, %4 : f64
      %6 = arith.addf %arg5, %5 : f64
      scf.yield %6 : f64
    }
    return %1 : f64
  }
  func.func @choose_cluster(%arg0: memref<?xf64>, %arg1: memref<?xmemref<?xf64>>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.7976931348623157E+308 : f64
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = call @approx_state_identity(%arg5) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c8_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = scf.index_switch %3 -> i32 
    case 0 {
      %5 = arith.index_cast %arg2 : i32 to index
      %6:2 = scf.for %arg6 = %c0 to %5 step %c1 iter_args(%arg7 = %c0_i32, %arg8 = %cst) -> (i32, f64) {
        %7 = arith.index_cast %arg6 : index to i32
        %8 = memref.load %arg1[%arg6] : memref<?xmemref<?xf64>>
        %9 = func.call @compute_distance_sq(%arg0, %8, %arg3, %arg4) : (memref<?xf64>, memref<?xf64>, i32, i32) -> f64
        %10 = arith.cmpf olt, %9, %arg8 : f64
        %11 = arith.select %10, %7, %arg7 : i32
        %12 = arith.select %10, %9, %arg8 : f64
        scf.yield %11, %12 : i32, f64
      }
      scf.yield %6#0 : i32
    }
    case 1 {
      %5 = arith.index_cast %arg2 : i32 to index
      %6:2 = scf.for %arg6 = %c0 to %5 step %c1 iter_args(%arg7 = %c0_i32, %arg8 = %cst) -> (i32, f64) {
        %7 = arith.index_cast %arg6 : index to i32
        %8 = memref.load %arg1[%arg6] : memref<?xmemref<?xf64>>
        %9 = func.call @compute_distance_sq(%arg0, %8, %arg3, %arg4) : (memref<?xf64>, memref<?xf64>, i32, i32) -> f64
        %10 = arith.cmpf olt, %9, %arg8 : f64
        %11 = arith.select %10, %7, %arg7 : i32
        %12 = arith.select %10, %9, %arg8 : f64
        scf.yield %11, %12 : i32, f64
      }
      scf.yield %6#0 : i32
    }
    default {
      %5 = arith.index_cast %arg2 : i32 to index
      %6:2 = scf.for %arg6 = %c0 to %5 step %c1 iter_args(%arg7 = %c0_i32, %arg8 = %cst) -> (i32, f64) {
        %7 = arith.index_cast %arg6 : index to i32
        %8 = memref.load %arg1[%arg6] : memref<?xmemref<?xf64>>
        %9 = func.call @compute_distance_sq(%arg0, %8, %arg3, %arg4) : (memref<?xf64>, memref<?xf64>, i32, i32) -> f64
        %10 = arith.cmpf olt, %9, %arg8 : f64
        %11 = arith.select %10, %7, %arg7 : i32
        %12 = arith.select %10, %9, %arg8 : f64
        scf.yield %11, %12 : i32, f64
      }
      scf.yield %6#0 : i32
    }
    return %4 : i32
  }
  func.func @reset_accumulators(%arg0: memref<?xmemref<?xf64>>, %arg1: memref<?xi32>, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    scf.for %arg4 = %c0 to %0 step %c1 {
      memref.store %c0_i32, %arg1[%arg4] : memref<?xi32>
      %1 = arith.index_cast %arg3 : i32 to index
      scf.for %arg5 = %c0 to %1 step %c1 {
        %2 = memref.load %arg0[%arg4] : memref<?xmemref<?xf64>>
        memref.store %cst, %2[%arg5] : memref<?xf64>
      }
    }
    return
  }
  func.func @assign_points_and_accumulate(%arg0: memref<?xmemref<?xf64>>, %arg1: memref<?xmemref<?xf64>>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xmemref<?xf64>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10000_i32 = arith.constant 10000 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = call @approx_state_identity(%arg9) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c10000_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.index_switch %3 
    case 0 {
      %4 = arith.index_cast %arg5 : i32 to index
      scf.for %arg10 = %c0 to %4 step %c1 {
        %5 = arith.index_cast %arg10 : index to i32
        %6 = memref.load %arg0[%arg10] : memref<?xmemref<?xf64>>
        %7 = func.call @choose_cluster(%6, %arg1, %arg7, %arg6, %5, %arg8) : (memref<?xf64>, memref<?xmemref<?xf64>>, i32, i32, i32, i32) -> i32
        memref.store %7, %arg2[%arg10] : memref<?xi32>
        %8 = arith.index_cast %7 : i32 to index
        %9 = memref.load %arg3[%8] : memref<?xi32>
        %10 = arith.addi %9, %c1_i32 : i32
        memref.store %10, %arg3[%8] : memref<?xi32>
        %11 = arith.index_cast %arg6 : i32 to index
        scf.for %arg11 = %c0 to %11 step %c1 {
          %12 = memref.load %arg4[%8] : memref<?xmemref<?xf64>>
          %13 = memref.load %arg0[%arg10] : memref<?xmemref<?xf64>>
          %14 = memref.load %13[%arg11] : memref<?xf64>
          %15 = memref.load %12[%arg11] : memref<?xf64>
          %16 = arith.addf %15, %14 : f64
          memref.store %16, %12[%arg11] : memref<?xf64>
        }
      }
      scf.yield
    }
    case 1 {
      %4 = arith.index_cast %arg5 : i32 to index
      scf.for %arg10 = %c0 to %4 step %c1 {
        %5 = arith.index_cast %arg10 : index to i32
        %6 = memref.load %arg0[%arg10] : memref<?xmemref<?xf64>>
        %7 = func.call @choose_cluster(%6, %arg1, %arg7, %arg6, %5, %arg8) : (memref<?xf64>, memref<?xmemref<?xf64>>, i32, i32, i32, i32) -> i32
        memref.store %7, %arg2[%arg10] : memref<?xi32>
        %8 = arith.index_cast %7 : i32 to index
        %9 = memref.load %arg3[%8] : memref<?xi32>
        %10 = arith.addi %9, %c1_i32 : i32
        memref.store %10, %arg3[%8] : memref<?xi32>
        %11 = arith.index_cast %arg6 : i32 to index
        scf.for %arg11 = %c0 to %11 step %c1 {
          %12 = memref.load %arg4[%8] : memref<?xmemref<?xf64>>
          %13 = memref.load %arg0[%arg10] : memref<?xmemref<?xf64>>
          %14 = memref.load %13[%arg11] : memref<?xf64>
          %15 = memref.load %12[%arg11] : memref<?xf64>
          %16 = arith.addf %15, %14 : f64
          memref.store %16, %12[%arg11] : memref<?xf64>
        }
      }
      scf.yield
    }
    default {
      %4 = arith.index_cast %arg5 : i32 to index
      scf.for %arg10 = %c0 to %4 step %c1 {
        %5 = arith.index_cast %arg10 : index to i32
        %6 = memref.load %arg0[%arg10] : memref<?xmemref<?xf64>>
        %7 = func.call @choose_cluster(%6, %arg1, %arg7, %arg6, %5, %arg8) : (memref<?xf64>, memref<?xmemref<?xf64>>, i32, i32, i32, i32) -> i32
        memref.store %7, %arg2[%arg10] : memref<?xi32>
        %8 = arith.index_cast %7 : i32 to index
        %9 = memref.load %arg3[%8] : memref<?xi32>
        %10 = arith.addi %9, %c1_i32 : i32
        memref.store %10, %arg3[%8] : memref<?xi32>
        %11 = arith.index_cast %arg6 : i32 to index
        scf.for %arg11 = %c0 to %11 step %c1 {
          %12 = memref.load %arg4[%8] : memref<?xmemref<?xf64>>
          %13 = memref.load %arg0[%arg10] : memref<?xmemref<?xf64>>
          %14 = memref.load %13[%arg11] : memref<?xf64>
          %15 = memref.load %12[%arg11] : memref<?xf64>
          %16 = arith.addf %15, %14 : f64
          memref.store %16, %12[%arg11] : memref<?xf64>
        }
      }
    }
    return
  }
  func.func @recompute_centroids(%arg0: memref<?xmemref<?xf64>>, %arg1: memref<?xmemref<?xf64>>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg3 : i32 to index
    scf.for %arg5 = %c0 to %0 step %c1 {
      %1 = memref.load %arg2[%arg5] : memref<?xi32>
      %2 = arith.cmpi sgt, %1, %c0_i32 : i32
      scf.if %2 {
        %3 = memref.load %arg2[%arg5] : memref<?xi32>
        %4 = arith.sitofp %3 : i32 to f64
        %5 = arith.divf %cst, %4 : f64
        %6 = arith.index_cast %arg4 : i32 to index
        scf.for %arg6 = %c0 to %6 step %c1 {
          %7 = memref.load %arg0[%arg5] : memref<?xmemref<?xf64>>
          %8 = memref.load %arg1[%arg5] : memref<?xmemref<?xf64>>
          %9 = memref.load %8[%arg6] : memref<?xf64>
          %10 = arith.mulf %9, %5 : f64
          memref.store %10, %7[%arg6] : memref<?xf64>
        }
      }
    }
    return
  }
  func.func @run_kmeans_iterations(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?xmemref<?xf64>>, %arg5: memref<?xmemref<?xf64>>, %arg6: memref<?xi32>, %arg7: memref<?xmemref<?xf64>>, %arg8: memref<?xi32>, %arg9: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10000_i32 = arith.constant 10000 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = call @approx_state_identity(%arg9) : (i32) -> i32
    %1 = arith.cmpi sge, %0, %c10000_i32 : i32
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.index_switch %3 
    case 0 {
      %4 = arith.index_cast %arg0 : i32 to index
      scf.for %arg10 = %c0 to %4 step %c1 {
        %5 = arith.index_cast %arg10 : index to i32
        func.call @reset_accumulators(%arg7, %arg8, %arg1, %arg2) : (memref<?xmemref<?xf64>>, memref<?xi32>, i32, i32) -> ()
        func.call @assign_points_and_accumulate(%arg4, %arg5, %arg6, %arg8, %arg7, %arg3, %arg2, %arg1, %5, %5) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, memref<?xi32>, memref<?xmemref<?xf64>>, i32, i32, i32, i32, i32) -> ()
        func.call @recompute_centroids(%arg5, %arg7, %arg8, %arg1, %arg2) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, i32, i32) -> ()
      }
      scf.yield
    }
    case 1 {
      %4 = arith.index_cast %arg0 : i32 to index
      scf.for %arg10 = %c0 to %4 step %c1 {
        %5 = arith.index_cast %arg10 : index to i32
        func.call @reset_accumulators(%arg7, %arg8, %arg1, %arg2) : (memref<?xmemref<?xf64>>, memref<?xi32>, i32, i32) -> ()
        func.call @assign_points_and_accumulate(%arg4, %arg5, %arg6, %arg8, %arg7, %arg3, %arg2, %arg1, %5, %5) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, memref<?xi32>, memref<?xmemref<?xf64>>, i32, i32, i32, i32, i32) -> ()
        func.call @recompute_centroids(%arg5, %arg7, %arg8, %arg1, %arg2) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, i32, i32) -> ()
      }
      scf.yield
    }
    default {
      %4 = arith.index_cast %arg0 : i32 to index
      scf.for %arg10 = %c0 to %4 step %c1 {
        %5 = arith.index_cast %arg10 : index to i32
        func.call @reset_accumulators(%arg7, %arg8, %arg1, %arg2) : (memref<?xmemref<?xf64>>, memref<?xi32>, i32, i32) -> ()
        func.call @assign_points_and_accumulate(%arg4, %arg5, %arg6, %arg8, %arg7, %arg3, %arg2, %arg1, %5, %5) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, memref<?xi32>, memref<?xmemref<?xf64>>, i32, i32, i32, i32, i32) -> ()
        func.call @recompute_centroids(%arg5, %arg7, %arg8, %arg1, %arg2) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, i32, i32) -> ()
      }
    }
    return
  }
  func.func @kmeans_kernel(%arg0: memref<?xmemref<?xf64>>, %arg1: memref<?xmemref<?xf64>>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %0 = arith.extsi %arg5 : i32 to i64
    %1 = arith.muli %0, %c4_i64 : i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = arith.divui %2, %c4 : index
    %alloc = memref.alloc(%3) : memref<?xi32>
    %4 = "polygeist.typeSize"() <{source = memref<?xf64>}> : () -> index
    %5 = arith.index_cast %4 : index to i64
    %6 = arith.muli %0, %5 : i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = arith.divui %7, %4 : index
    %alloc_0 = memref.alloc(%8) : memref<?xmemref<?xf64>>
    %9 = arith.index_cast %arg5 : i32 to index
    scf.for %arg7 = %c0 to %9 step %c1 {
      %11 = arith.extsi %arg4 : i32 to i64
      %12 = arith.muli %11, %c8_i64 : i64
      %13 = arith.index_cast %12 : i64 to index
      %14 = arith.divui %13, %c8 : index
      %alloc_1 = memref.alloc(%14) : memref<?xf64>
      memref.store %alloc_1, %alloc_0[%arg7] : memref<?xmemref<?xf64>>
    }
    call @run_kmeans_iterations(%arg6, %arg5, %arg4, %arg3, %arg0, %arg1, %arg2, %alloc_0, %alloc, %arg3) : (i32, i32, i32, i32, memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, memref<?xmemref<?xf64>>, memref<?xi32>, i32) -> ()
    %10 = arith.index_cast %arg5 : i32 to index
    scf.for %arg7 = %c0 to %10 step %c1 {
      %11 = memref.load %alloc_0[%arg7] : memref<?xmemref<?xf64>>
      memref.dealloc %11 : memref<?xf64>
    }
    memref.dealloc %alloc_0 : memref<?xmemref<?xf64>>
    memref.dealloc %alloc : memref<?xi32>
    return
  }
  func.func @generate_random_data(%arg0: memref<?xmemref<?xf64>>, %arg1: memref<?xmemref<?xf64>>, %arg2: i32, %arg3: i32, %arg4: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+02 : f64
    %cst_0 = arith.constant 0x41DFFFFFFFC00000 : f64
    %0 = memref.get_global @seed : memref<1xi32>
    %1 = affine.load %0[0] : memref<1xi32>
    call @srand(%1) : (i32) -> ()
    %2 = arith.index_cast %arg2 : i32 to index
    scf.for %arg5 = %c0 to %2 step %c1 {
      %4 = arith.index_cast %arg3 : i32 to index
      scf.for %arg6 = %c0 to %4 step %c1 {
        %5 = memref.load %arg0[%arg5] : memref<?xmemref<?xf64>>
        %6 = func.call @rand() : () -> i32
        %7 = arith.sitofp %6 : i32 to f64
        %8 = arith.divf %7, %cst_0 : f64
        %9 = arith.mulf %8, %cst : f64
        memref.store %9, %5[%arg6] : memref<?xf64>
      }
    }
    %3 = arith.index_cast %arg4 : i32 to index
    scf.for %arg5 = %c0 to %3 step %c1 {
      %4 = func.call @rand() : () -> i32
      %5 = arith.remsi %4, %arg2 : i32
      %6 = arith.index_cast %arg3 : i32 to index
      scf.for %arg6 = %c0 to %6 step %c1 {
        %7 = memref.load %arg1[%arg5] : memref<?xmemref<?xf64>>
        %8 = arith.index_cast %5 : i32 to index
        %9 = memref.load %arg0[%8] : memref<?xmemref<?xf64>>
        %10 = memref.load %9[%arg6] : memref<?xf64>
        memref.store %10, %7[%arg6] : memref<?xf64>
      }
    }
    return
  }
  func.func private @srand(i32) attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @rand() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @print_results(%arg0: memref<?xmemref<?xf64>>, %arg1: memref<?xmemref<?xf64>>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.addressof @str0 : !llvm.ptr
    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<18 x i8>
    %3 = llvm.call @printf(%2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %4 = arith.index_cast %arg5 : i32 to index
    scf.for %arg6 = %c0 to %4 step %c1 {
      %20 = arith.index_cast %arg6 : index to i32
      %21 = llvm.mlir.addressof @str1 : !llvm.ptr
      %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<15 x i8>
      %23 = llvm.call @printf(%22, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %24 = arith.index_cast %arg4 : i32 to index
      scf.for %arg7 = %c0 to %24 step %c1 {
        %28 = arith.index_cast %arg7 : index to i32
        %29 = llvm.mlir.addressof @str2 : !llvm.ptr
        %30 = llvm.getelementptr %29[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
        %31 = memref.load %arg1[%arg6] : memref<?xmemref<?xf64>>
        %32 = memref.load %31[%arg7] : memref<?xf64>
        %33 = llvm.call @printf(%30, %32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
        %34 = arith.addi %arg4, %c-1_i32 : i32
        %35 = arith.cmpi slt, %28, %34 : i32
        scf.if %35 {
          %36 = llvm.mlir.addressof @str3 : !llvm.ptr
          %37 = llvm.getelementptr %36[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
          %38 = llvm.call @printf(%37) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        }
      }
      %25 = llvm.mlir.addressof @str4 : !llvm.ptr
      %26 = llvm.getelementptr %25[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
      %27 = llvm.call @printf(%26) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }
    %5 = arith.extsi %arg5 : i32 to i64
    %6 = arith.muli %5, %c4_i64 : i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = arith.divui %7, %c4 : index
    %alloc = memref.alloc(%8) : memref<?xi32>
    %9 = arith.index_cast %arg5 : i32 to index
    scf.for %arg6 = %c0 to %9 step %c1 {
      memref.store %c0_i32, %alloc[%arg6] : memref<?xi32>
    }
    %10 = arith.index_cast %arg3 : i32 to index
    scf.for %arg6 = %c0 to %10 step %c1 {
      %20 = memref.load %arg2[%arg6] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %alloc[%21] : memref<?xi32>
      %23 = arith.addi %22, %c1_i32 : i32
      memref.store %23, %alloc[%21] : memref<?xi32>
    }
    %11 = llvm.mlir.addressof @str5 : !llvm.ptr
    %12 = llvm.getelementptr %11[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    %13 = llvm.call @printf(%12) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %14 = arith.index_cast %arg5 : i32 to index
    scf.for %arg6 = %c0 to %14 step %c1 {
      %20 = arith.index_cast %arg6 : index to i32
      %21 = llvm.mlir.addressof @str6 : !llvm.ptr
      %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
      %23 = memref.load %alloc[%arg6] : memref<?xi32>
      %24 = llvm.call @printf(%22, %20, %23) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
    }
    %15 = llvm.mlir.addressof @str7 : !llvm.ptr
    %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<35 x i8>
    %17 = llvm.call @printf(%16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %18 = arith.index_cast %arg5 : i32 to index
    %19 = scf.for %arg6 = %c0 to %18 step %c1 iter_args(%arg7 = %0) -> (i32) {
      %20 = arith.index_cast %arg6 : index to i32
      %21 = llvm.mlir.addressof @str8 : !llvm.ptr
      %22 = llvm.getelementptr %21[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<21 x i8>
      %23 = llvm.call @printf(%22, %20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %24:3 = scf.while (%arg8 = %arg7, %arg9 = %c0_i32, %arg10 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
        %25 = arith.cmpi slt, %arg9, %arg3 : i32
        %26 = arith.cmpi slt, %arg10, %c3_i32 : i32
        %27 = arith.andi %25, %26 : i1
        scf.condition(%27) %arg8, %arg9, %arg10 : i32, i32, i32
      } do {
      ^bb0(%arg8: i32, %arg9: i32, %arg10: i32):
        %25 = arith.index_cast %arg9 : i32 to index
        %26 = memref.load %arg2[%25] : memref<?xi32>
        %27 = arith.cmpi eq, %26, %20 : i32
        %28 = arith.select %27, %arg4, %arg8 : i32
        %29 = scf.if %27 -> (i32) {
          %31 = llvm.mlir.addressof @str9 : !llvm.ptr
          %32 = llvm.getelementptr %31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
          %33 = llvm.call @printf(%32, %arg9) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
          %34 = arith.index_cast %arg4 : i32 to index
          scf.for %arg11 = %c0 to %34 step %c1 {
            %39 = arith.index_cast %arg11 : index to i32
            %40 = llvm.mlir.addressof @str2 : !llvm.ptr
            %41 = llvm.getelementptr %40[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
            %42 = memref.load %arg0[%25] : memref<?xmemref<?xf64>>
            %43 = memref.load %42[%arg11] : memref<?xf64>
            %44 = llvm.call @printf(%41, %43) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
            %45 = arith.addi %arg4, %c-1_i32 : i32
            %46 = arith.cmpi slt, %39, %45 : i32
            scf.if %46 {
              %47 = llvm.mlir.addressof @str3 : !llvm.ptr
              %48 = llvm.getelementptr %47[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
              %49 = llvm.call @printf(%48) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
            }
          }
          %35 = llvm.mlir.addressof @str4 : !llvm.ptr
          %36 = llvm.getelementptr %35[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
          %37 = llvm.call @printf(%36) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
          %38 = arith.addi %arg10, %c1_i32 : i32
          scf.yield %38 : i32
        } else {
          scf.yield %arg10 : i32
        }
        %30 = arith.addi %arg9, %c1_i32 : i32
        scf.yield %28, %30, %29 : i32, i32, i32
      }
      scf.yield %24#0 : i32
    }
    memref.dealloc %alloc : memref<?xi32>
    return
  }
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c8_i64 = arith.constant 8 : i64
    %c4 = arith.constant 4 : index
    %c4_i64 = arith.constant 4 : i64
    %cst = arith.constant 1.000000e+06 : f64
    %cst_0 = arith.constant 1.000000e+03 : f64
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c20_i32 = arith.constant 20 : i32
    %c5_i32 = arith.constant 5 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<1x2xi64>
    %alloca_1 = memref.alloca() : memref<1x2xi64>
    %1:7 = scf.while (%arg2 = %c1_i32, %arg3 = %c20_i32, %arg4 = %c5_i32, %arg5 = %c2_i32, %arg6 = %c1000_i32, %arg7 = %true, %arg8 = %0, %arg9 = %true) : (i32, i32, i32, i32, i32, i1, i32, i1) -> (i32, i32, i32, i32, i1, i32, i32) {
      %3 = arith.cmpi slt, %arg2, %arg0 : i32
      %4 = arith.andi %3, %arg9 : i1
      scf.condition(%4) %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg2 : i32, i32, i32, i32, i1, i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i1, %arg7: i32, %arg8: i32):
      %3 = arith.index_cast %arg8 : i32 to index
      %4 = memref.load %arg1[%3] : memref<?xmemref<?xi8>>
      %5 = llvm.mlir.addressof @str10 : !llvm.ptr
      %6 = "polygeist.pointer2memref"(%5) : (!llvm.ptr) -> memref<?xi8>
      %7 = func.call @strcmp(%4, %6) : (memref<?xi8>, memref<?xi8>) -> i32
      %8 = arith.cmpi eq, %7, %c0_i32 : i32
      %9 = scf.if %8 -> (i1) {
        %12 = arith.addi %arg8, %c1_i32 : i32
        %13 = arith.cmpi slt, %12, %arg0 : i32
        scf.yield %13 : i1
      } else {
        scf.yield %false : i1
      }
      %10:8 = scf.if %9 -> (i32, i32, i32, i32, i32, i1, i32, i1) {
        %12 = arith.addi %arg8, %c1_i32 : i32
        %13 = arith.index_cast %12 : i32 to index
        %14 = memref.load %arg1[%13] : memref<?xmemref<?xi8>>
        %15 = func.call @atoi(%14) : (memref<?xi8>) -> i32
        %16 = arith.addi %arg8, %c1_i32 : i32
        scf.yield %16, %arg2, %arg3, %arg4, %15, %arg6, %arg7, %true : i32, i32, i32, i32, i32, i1, i32, i1
      } else {
        %12 = memref.load %arg1[%3] : memref<?xmemref<?xi8>>
        %13 = llvm.mlir.addressof @str11 : !llvm.ptr
        %14 = "polygeist.pointer2memref"(%13) : (!llvm.ptr) -> memref<?xi8>
        %15 = func.call @strcmp(%12, %14) : (memref<?xi8>, memref<?xi8>) -> i32
        %16 = arith.cmpi eq, %15, %c0_i32 : i32
        %17 = scf.if %16 -> (i1) {
          %19 = arith.addi %arg8, %c1_i32 : i32
          %20 = arith.cmpi slt, %19, %arg0 : i32
          scf.yield %20 : i1
        } else {
          scf.yield %false : i1
        }
        %18:7 = scf.if %17 -> (i32, i32, i32, i32, i1, i32, i1) {
          %19 = arith.addi %arg8, %c1_i32 : i32
          %20 = arith.index_cast %19 : i32 to index
          %21 = memref.load %arg1[%20] : memref<?xmemref<?xi8>>
          %22 = func.call @atoi(%21) : (memref<?xi8>) -> i32
          %23 = arith.addi %arg8, %c1_i32 : i32
          scf.yield %23, %arg2, %arg3, %22, %arg6, %arg7, %true : i32, i32, i32, i32, i1, i32, i1
        } else {
          %19 = memref.load %arg1[%3] : memref<?xmemref<?xi8>>
          %20 = llvm.mlir.addressof @str12 : !llvm.ptr
          %21 = "polygeist.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi8>
          %22 = func.call @strcmp(%19, %21) : (memref<?xi8>, memref<?xi8>) -> i32
          %23 = arith.cmpi eq, %22, %c0_i32 : i32
          %24 = scf.if %23 -> (i1) {
            %26 = arith.addi %arg8, %c1_i32 : i32
            %27 = arith.cmpi slt, %26, %arg0 : i32
            scf.yield %27 : i1
          } else {
            scf.yield %false : i1
          }
          %25:6 = scf.if %24 -> (i32, i32, i32, i1, i32, i1) {
            %26 = arith.addi %arg8, %c1_i32 : i32
            %27 = arith.index_cast %26 : i32 to index
            %28 = memref.load %arg1[%27] : memref<?xmemref<?xi8>>
            %29 = func.call @atoi(%28) : (memref<?xi8>) -> i32
            %30 = arith.addi %arg8, %c1_i32 : i32
            scf.yield %30, %arg2, %29, %arg6, %arg7, %true : i32, i32, i32, i1, i32, i1
          } else {
            %26 = memref.load %arg1[%3] : memref<?xmemref<?xi8>>
            %27 = llvm.mlir.addressof @str13 : !llvm.ptr
            %28 = "polygeist.pointer2memref"(%27) : (!llvm.ptr) -> memref<?xi8>
            %29 = func.call @strcmp(%26, %28) : (memref<?xi8>, memref<?xi8>) -> i32
            %30 = arith.cmpi eq, %29, %c0_i32 : i32
            %31 = scf.if %30 -> (i1) {
              %33 = arith.addi %arg8, %c1_i32 : i32
              %34 = arith.cmpi slt, %33, %arg0 : i32
              scf.yield %34 : i1
            } else {
              scf.yield %false : i1
            }
            %32:5 = scf.if %31 -> (i32, i32, i1, i32, i1) {
              %33 = arith.addi %arg8, %c1_i32 : i32
              %34 = arith.index_cast %33 : i32 to index
              %35 = memref.load %arg1[%34] : memref<?xmemref<?xi8>>
              %36 = func.call @atoi(%35) : (memref<?xi8>) -> i32
              %37 = arith.addi %arg8, %c1_i32 : i32
              scf.yield %37, %36, %arg6, %arg7, %true : i32, i32, i1, i32, i1
            } else {
              %33 = memref.load %arg1[%3] : memref<?xmemref<?xi8>>
              %34 = llvm.mlir.addressof @str14 : !llvm.ptr
              %35 = "polygeist.pointer2memref"(%34) : (!llvm.ptr) -> memref<?xi8>
              %36 = func.call @strcmp(%33, %35) : (memref<?xi8>, memref<?xi8>) -> i32
              %37 = arith.cmpi eq, %36, %c0_i32 : i32
              %38 = scf.if %37 -> (i1) {
                %40 = arith.addi %arg8, %c1_i32 : i32
                %41 = arith.cmpi slt, %40, %arg0 : i32
                scf.yield %41 : i1
              } else {
                scf.yield %false : i1
              }
              %39:4 = scf.if %38 -> (i32, i1, i32, i1) {
                %40 = memref.get_global @seed : memref<1xi32>
                %41 = arith.addi %arg8, %c1_i32 : i32
                %42 = arith.index_cast %41 : i32 to index
                %43 = memref.load %arg1[%42] : memref<?xmemref<?xi8>>
                %44 = func.call @atoi(%43) : (memref<?xi8>) -> i32
                affine.store %44, %40[0] : memref<1xi32>
                %45 = arith.addi %arg8, %c1_i32 : i32
                scf.yield %45, %arg6, %arg7, %true : i32, i1, i32, i1
              } else {
                %40 = memref.load %arg1[%3] : memref<?xmemref<?xi8>>
                %41 = llvm.mlir.addressof @str15 : !llvm.ptr
                %42 = "polygeist.pointer2memref"(%41) : (!llvm.ptr) -> memref<?xi8>
                %43 = func.call @strcmp(%40, %42) : (memref<?xi8>, memref<?xi8>) -> i32
                %44 = arith.cmpi eq, %43, %c0_i32 : i32
                %45 = arith.cmpi ne, %43, %c0_i32 : i32
                %46 = arith.andi %45, %arg6 : i1
                %47 = arith.select %44, %c0_i32, %arg7 : i32
                %48 = arith.cmpi ne, %43, %c0_i32 : i32
                scf.if %44 {
                  %49 = llvm.mlir.addressof @str16 : !llvm.ptr
                  %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<77 x i8>
                  %51 = affine.load %arg1[0] : memref<?xmemref<?xi8>>
                  %52 = "polygeist.memref2pointer"(%51) : (memref<?xi8>) -> !llvm.ptr
                  %53 = llvm.call @printf(%50, %52) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
                }
                scf.yield %arg8, %46, %47, %48 : i32, i1, i32, i1
              }
              scf.yield %39#0, %arg2, %39#1, %39#2, %39#3 : i32, i32, i1, i32, i1
            }
            scf.yield %32#0, %32#1, %arg3, %32#2, %32#3, %32#4 : i32, i32, i32, i1, i32, i1
          }
          scf.yield %25#0, %25#1, %25#2, %arg4, %25#3, %25#4, %25#5 : i32, i32, i32, i32, i1, i32, i1
        }
        scf.yield %18#0, %18#1, %18#2, %18#3, %arg5, %18#4, %18#5, %18#6 : i32, i32, i32, i32, i32, i1, i32, i1
      }
      %11 = scf.if %10#7 -> (i32) {
        %12 = arith.addi %10#0, %c1_i32 : i32
        scf.yield %12 : i32
      } else {
        scf.yield %arg8 : i32
      }
      scf.yield %11, %10#1, %10#2, %10#3, %10#4, %10#5, %10#6, %10#7 : i32, i32, i32, i32, i32, i1, i32, i1
    }
    %2 = arith.select %1#4, %c0_i32, %1#5 : i32
    scf.if %1#4 {
      %3 = llvm.mlir.addressof @str17 : !llvm.ptr
      %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<23 x i8>
      %5 = llvm.call @printf(%4) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      %6 = llvm.mlir.addressof @str18 : !llvm.ptr
      %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<14 x i8>
      %8 = llvm.call @printf(%7, %1#3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %9 = llvm.mlir.addressof @str19 : !llvm.ptr
      %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<18 x i8>
      %11 = llvm.call @printf(%10, %1#2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %12 = llvm.mlir.addressof @str20 : !llvm.ptr
      %13 = llvm.getelementptr %12[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i8>
      %14 = llvm.call @printf(%13, %1#1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %15 = llvm.mlir.addressof @str21 : !llvm.ptr
      %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
      %17 = llvm.call @printf(%16, %1#0) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      %18 = "polygeist.typeSize"() <{source = memref<?xf64>}> : () -> index
      %19 = arith.extsi %1#3 : i32 to i64
      %20 = arith.index_cast %18 : index to i64
      %21 = arith.muli %19, %20 : i64
      %22 = arith.index_cast %21 : i64 to index
      %23 = arith.divui %22, %18 : index
      %alloc = memref.alloc(%23) : memref<?xmemref<?xf64>>
      %24 = arith.index_cast %1#3 : i32 to index
      scf.for %arg2 = %c0 to %24 step %c1 {
        %54 = arith.extsi %1#2 : i32 to i64
        %55 = arith.muli %54, %c8_i64 : i64
        %56 = arith.index_cast %55 : i64 to index
        %57 = arith.divui %56, %c8 : index
        %alloc_5 = memref.alloc(%57) : memref<?xf64>
        memref.store %alloc_5, %alloc[%arg2] : memref<?xmemref<?xf64>>
      }
      %25 = "polygeist.typeSize"() <{source = memref<?xf64>}> : () -> index
      %26 = arith.extsi %1#1 : i32 to i64
      %27 = arith.index_cast %25 : index to i64
      %28 = arith.muli %26, %27 : i64
      %29 = arith.index_cast %28 : i64 to index
      %30 = arith.divui %29, %25 : index
      %alloc_2 = memref.alloc(%30) : memref<?xmemref<?xf64>>
      %31 = arith.index_cast %1#1 : i32 to index
      scf.for %arg2 = %c0 to %31 step %c1 {
        %54 = arith.extsi %1#2 : i32 to i64
        %55 = arith.muli %54, %c8_i64 : i64
        %56 = arith.index_cast %55 : i64 to index
        %57 = arith.divui %56, %c8 : index
        %alloc_5 = memref.alloc(%57) : memref<?xf64>
        memref.store %alloc_5, %alloc_2[%arg2] : memref<?xmemref<?xf64>>
      }
      %32 = arith.extsi %1#3 : i32 to i64
      %33 = arith.muli %32, %c4_i64 : i64
      %34 = arith.index_cast %33 : i64 to index
      %35 = arith.divui %34, %c4 : index
      %alloc_3 = memref.alloc(%35) : memref<?xi32>
      func.call @generate_random_data(%alloc, %alloc_2, %1#3, %1#2, %1#1) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, i32, i32, i32) -> ()
      %cast = memref.cast %alloca_1 : memref<1x2xi64> to memref<?x2xi64>
      %36 = func.call @clock_gettime(%c1_i32, %cast) : (i32, memref<?x2xi64>) -> i32
      func.call @kmeans_kernel(%alloc, %alloc_2, %alloc_3, %1#3, %1#2, %1#1, %1#0) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, i32, i32, i32, i32) -> ()
      %cast_4 = memref.cast %alloca : memref<1x2xi64> to memref<?x2xi64>
      %37 = func.call @clock_gettime(%c1_i32, %cast_4) : (i32, memref<?x2xi64>) -> i32
      %38 = affine.load %alloca[0, 0] : memref<1x2xi64>
      %39 = affine.load %alloca_1[0, 0] : memref<1x2xi64>
      %40 = arith.subi %38, %39 : i64
      %41 = arith.sitofp %40 : i64 to f64
      %42 = arith.mulf %41, %cst_0 : f64
      %43 = affine.load %alloca[0, 1] : memref<1x2xi64>
      %44 = affine.load %alloca_1[0, 1] : memref<1x2xi64>
      %45 = arith.subi %43, %44 : i64
      %46 = arith.sitofp %45 : i64 to f64
      %47 = arith.divf %46, %cst : f64
      %48 = arith.addf %42, %47 : f64
      %49 = llvm.mlir.addressof @str22 : !llvm.ptr
      %50 = llvm.getelementptr %49[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<31 x i8>
      %51 = llvm.call @printf(%50, %48) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      func.call @print_results(%alloc, %alloc_2, %alloc_3, %1#3, %1#2, %1#1) : (memref<?xmemref<?xf64>>, memref<?xmemref<?xf64>>, memref<?xi32>, i32, i32, i32) -> ()
      %52 = arith.index_cast %1#3 : i32 to index
      scf.for %arg2 = %c0 to %52 step %c1 {
        %54 = memref.load %alloc[%arg2] : memref<?xmemref<?xf64>>
        memref.dealloc %54 : memref<?xf64>
      }
      memref.dealloc %alloc : memref<?xmemref<?xf64>>
      %53 = arith.index_cast %1#1 : i32 to index
      scf.for %arg2 = %c0 to %53 step %c1 {
        %54 = memref.load %alloc_2[%arg2] : memref<?xmemref<?xf64>>
        memref.dealloc %54 : memref<?xf64>
      }
      memref.dealloc %alloc_2 : memref<?xmemref<?xf64>>
      memref.dealloc %alloc_3 : memref<?xi32>
    }
    return %2 : i32
  }
  func.func private @strcmp(memref<?xi8>, memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @atoi(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @clock_gettime(i32, memref<?x2xi64>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}

