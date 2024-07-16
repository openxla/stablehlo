// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf16>
    %1 = call @expected() : () -> tensor<20x20xf16>
    %2 = stablehlo.convert %0 : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %3 = stablehlo.abs %2 : tensor<20x20xf32>
    %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_1 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %7 = stablehlo.multiply %4, %3 : tensor<20x20xf32>
    %8 = stablehlo.subtract %7, %5 : tensor<20x20xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %12 = stablehlo.multiply %8, %9 : tensor<20x20xf32>
    %13 = stablehlo.subtract %12, %10 : tensor<20x20xf32>
    %cst_5 = stablehlo.constant dense<-1.300025009986248E-8> : tensor<f64>
    %14 = stablehlo.convert %cst_5 : (tensor<f64>) -> tensor<f32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %16 = stablehlo.add %13, %15 : tensor<20x20xf32>
    %17 = stablehlo.multiply %8, %16 : tensor<20x20xf32>
    %18 = stablehlo.subtract %17, %9 : tensor<20x20xf32>
    %cst_6 = stablehlo.constant dense<6.0469950225419186E-8> : tensor<f64>
    %19 = stablehlo.convert %cst_6 : (tensor<f64>) -> tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %21 = stablehlo.add %18, %20 : tensor<20x20xf32>
    %22 = stablehlo.multiply %8, %21 : tensor<20x20xf32>
    %23 = stablehlo.subtract %22, %16 : tensor<20x20xf32>
    %cst_7 = stablehlo.constant dense<-2.6707938539406119E-7> : tensor<f64>
    %24 = stablehlo.convert %cst_7 : (tensor<f64>) -> tensor<f32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %26 = stablehlo.add %23, %25 : tensor<20x20xf32>
    %27 = stablehlo.multiply %8, %26 : tensor<20x20xf32>
    %28 = stablehlo.subtract %27, %21 : tensor<20x20xf32>
    %cst_8 = stablehlo.constant dense<1.1173875391201037E-6> : tensor<f64>
    %29 = stablehlo.convert %cst_8 : (tensor<f64>) -> tensor<f32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %31 = stablehlo.add %28, %30 : tensor<20x20xf32>
    %32 = stablehlo.multiply %8, %31 : tensor<20x20xf32>
    %33 = stablehlo.subtract %32, %26 : tensor<20x20xf32>
    %cst_9 = stablehlo.constant dense<-4.4167383584587505E-6> : tensor<f64>
    %34 = stablehlo.convert %cst_9 : (tensor<f64>) -> tensor<f32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %36 = stablehlo.add %33, %35 : tensor<20x20xf32>
    %37 = stablehlo.multiply %8, %36 : tensor<20x20xf32>
    %38 = stablehlo.subtract %37, %31 : tensor<20x20xf32>
    %cst_10 = stablehlo.constant dense<1.6448448070728896E-5> : tensor<f64>
    %39 = stablehlo.convert %cst_10 : (tensor<f64>) -> tensor<f32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %41 = stablehlo.add %38, %40 : tensor<20x20xf32>
    %42 = stablehlo.multiply %8, %41 : tensor<20x20xf32>
    %43 = stablehlo.subtract %42, %36 : tensor<20x20xf32>
    %cst_11 = stablehlo.constant dense<-5.754195010082104E-5> : tensor<f64>
    %44 = stablehlo.convert %cst_11 : (tensor<f64>) -> tensor<f32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %46 = stablehlo.add %43, %45 : tensor<20x20xf32>
    %47 = stablehlo.multiply %8, %46 : tensor<20x20xf32>
    %48 = stablehlo.subtract %47, %41 : tensor<20x20xf32>
    %cst_12 = stablehlo.constant dense<1.8850288509584165E-4> : tensor<f64>
    %49 = stablehlo.convert %cst_12 : (tensor<f64>) -> tensor<f32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %51 = stablehlo.add %48, %50 : tensor<20x20xf32>
    %52 = stablehlo.multiply %8, %51 : tensor<20x20xf32>
    %53 = stablehlo.subtract %52, %46 : tensor<20x20xf32>
    %cst_13 = stablehlo.constant dense<-5.7637557453858236E-4> : tensor<f64>
    %54 = stablehlo.convert %cst_13 : (tensor<f64>) -> tensor<f32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %56 = stablehlo.add %53, %55 : tensor<20x20xf32>
    %57 = stablehlo.multiply %8, %56 : tensor<20x20xf32>
    %58 = stablehlo.subtract %57, %51 : tensor<20x20xf32>
    %cst_14 = stablehlo.constant dense<0.0016394756169413357> : tensor<f64>
    %59 = stablehlo.convert %cst_14 : (tensor<f64>) -> tensor<f32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %61 = stablehlo.add %58, %60 : tensor<20x20xf32>
    %62 = stablehlo.multiply %8, %61 : tensor<20x20xf32>
    %63 = stablehlo.subtract %62, %56 : tensor<20x20xf32>
    %cst_15 = stablehlo.constant dense<-0.0043243099950505759> : tensor<f64>
    %64 = stablehlo.convert %cst_15 : (tensor<f64>) -> tensor<f32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %66 = stablehlo.add %63, %65 : tensor<20x20xf32>
    %67 = stablehlo.multiply %8, %66 : tensor<20x20xf32>
    %68 = stablehlo.subtract %67, %61 : tensor<20x20xf32>
    %cst_16 = stablehlo.constant dense<0.010546460394594998> : tensor<f64>
    %69 = stablehlo.convert %cst_16 : (tensor<f64>) -> tensor<f32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %71 = stablehlo.add %68, %70 : tensor<20x20xf32>
    %72 = stablehlo.multiply %8, %71 : tensor<20x20xf32>
    %73 = stablehlo.subtract %72, %66 : tensor<20x20xf32>
    %cst_17 = stablehlo.constant dense<-0.023737414805899471> : tensor<f64>
    %74 = stablehlo.convert %cst_17 : (tensor<f64>) -> tensor<f32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %76 = stablehlo.add %73, %75 : tensor<20x20xf32>
    %77 = stablehlo.multiply %8, %76 : tensor<20x20xf32>
    %78 = stablehlo.subtract %77, %71 : tensor<20x20xf32>
    %cst_18 = stablehlo.constant dense<0.049305284239670712> : tensor<f64>
    %79 = stablehlo.convert %cst_18 : (tensor<f64>) -> tensor<f32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %81 = stablehlo.add %78, %80 : tensor<20x20xf32>
    %82 = stablehlo.multiply %8, %81 : tensor<20x20xf32>
    %83 = stablehlo.subtract %82, %76 : tensor<20x20xf32>
    %cst_19 = stablehlo.constant dense<-0.094901097048047639> : tensor<f64>
    %84 = stablehlo.convert %cst_19 : (tensor<f64>) -> tensor<f32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %86 = stablehlo.add %83, %85 : tensor<20x20xf32>
    %87 = stablehlo.multiply %8, %86 : tensor<20x20xf32>
    %88 = stablehlo.subtract %87, %81 : tensor<20x20xf32>
    %cst_20 = stablehlo.constant dense<0.17162090152220877> : tensor<f64>
    %89 = stablehlo.convert %cst_20 : (tensor<f64>) -> tensor<f32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %91 = stablehlo.add %88, %90 : tensor<20x20xf32>
    %92 = stablehlo.multiply %8, %91 : tensor<20x20xf32>
    %93 = stablehlo.subtract %92, %86 : tensor<20x20xf32>
    %cst_21 = stablehlo.constant dense<-0.3046826723431984> : tensor<f64>
    %94 = stablehlo.convert %cst_21 : (tensor<f64>) -> tensor<f32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %96 = stablehlo.add %93, %95 : tensor<20x20xf32>
    %97 = stablehlo.multiply %8, %96 : tensor<20x20xf32>
    %98 = stablehlo.subtract %97, %91 : tensor<20x20xf32>
    %cst_22 = stablehlo.constant dense<0.67679527440947607> : tensor<f64>
    %99 = stablehlo.convert %cst_22 : (tensor<f64>) -> tensor<f32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %101 = stablehlo.add %98, %100 : tensor<20x20xf32>
    %102 = stablehlo.subtract %101, %91 : tensor<20x20xf32>
    %cst_23 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %104 = stablehlo.multiply %103, %102 : tensor<20x20xf32>
    %105 = stablehlo.divide %6, %3 : tensor<20x20xf32>
    %106 = stablehlo.subtract %105, %5 : tensor<20x20xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %107 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %108 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %109 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %110 = stablehlo.multiply %106, %107 : tensor<20x20xf32>
    %111 = stablehlo.subtract %110, %108 : tensor<20x20xf32>
    %cst_27 = stablehlo.constant dense<3.3962320257083865E-9> : tensor<f64>
    %112 = stablehlo.convert %cst_27 : (tensor<f64>) -> tensor<f32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %114 = stablehlo.add %111, %113 : tensor<20x20xf32>
    %115 = stablehlo.multiply %106, %114 : tensor<20x20xf32>
    %116 = stablehlo.subtract %115, %107 : tensor<20x20xf32>
    %cst_28 = stablehlo.constant dense<2.266668990498178E-8> : tensor<f64>
    %117 = stablehlo.convert %cst_28 : (tensor<f64>) -> tensor<f32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %119 = stablehlo.add %116, %118 : tensor<20x20xf32>
    %120 = stablehlo.multiply %106, %119 : tensor<20x20xf32>
    %121 = stablehlo.subtract %120, %114 : tensor<20x20xf32>
    %cst_29 = stablehlo.constant dense<2.0489185894690638E-7> : tensor<f64>
    %122 = stablehlo.convert %cst_29 : (tensor<f64>) -> tensor<f32>
    %123 = stablehlo.broadcast_in_dim %122, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %124 = stablehlo.add %121, %123 : tensor<20x20xf32>
    %125 = stablehlo.multiply %106, %124 : tensor<20x20xf32>
    %126 = stablehlo.subtract %125, %119 : tensor<20x20xf32>
    %cst_30 = stablehlo.constant dense<2.8913705208347567E-6> : tensor<f64>
    %127 = stablehlo.convert %cst_30 : (tensor<f64>) -> tensor<f32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %129 = stablehlo.add %126, %128 : tensor<20x20xf32>
    %130 = stablehlo.multiply %106, %129 : tensor<20x20xf32>
    %131 = stablehlo.subtract %130, %124 : tensor<20x20xf32>
    %cst_31 = stablehlo.constant dense<6.8897583469168245E-5> : tensor<f64>
    %132 = stablehlo.convert %cst_31 : (tensor<f64>) -> tensor<f32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %134 = stablehlo.add %131, %133 : tensor<20x20xf32>
    %135 = stablehlo.multiply %106, %134 : tensor<20x20xf32>
    %136 = stablehlo.subtract %135, %129 : tensor<20x20xf32>
    %cst_32 = stablehlo.constant dense<0.0033691164782556943> : tensor<f64>
    %137 = stablehlo.convert %cst_32 : (tensor<f64>) -> tensor<f32>
    %138 = stablehlo.broadcast_in_dim %137, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %139 = stablehlo.add %136, %138 : tensor<20x20xf32>
    %140 = stablehlo.multiply %106, %139 : tensor<20x20xf32>
    %141 = stablehlo.subtract %140, %134 : tensor<20x20xf32>
    %cst_33 = stablehlo.constant dense<0.80449041101410879> : tensor<f64>
    %142 = stablehlo.convert %cst_33 : (tensor<f64>) -> tensor<f32>
    %143 = stablehlo.broadcast_in_dim %142, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %144 = stablehlo.add %141, %143 : tensor<20x20xf32>
    %145 = stablehlo.subtract %144, %134 : tensor<20x20xf32>
    %cst_34 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %146 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %147 = stablehlo.multiply %146, %145 : tensor<20x20xf32>
    %148 = stablehlo.sqrt %3 : tensor<20x20xf32>
    %149 = stablehlo.divide %147, %148 : tensor<20x20xf32>
    %cst_35 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %150 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %151 = stablehlo.compare  LE, %3, %150,  FLOAT : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %152 = stablehlo.select %151, %104, %149 : tensor<20x20xi1>, tensor<20x20xf32>
    %153 = stablehlo.convert %152 : (tensor<20x20xf32>) -> tensor<20x20xf16>
    stablehlo.custom_call @check.expect_close(%153, %1) {has_side_effect = true} : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    return %153 : tensor<20x20xf16>
  }
  func.func private @inputs() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x913768BC07BE0D38E6B8D8BDFAC403B078B859441FC21BBCDBBC7141DB40184276C7E9C69BBB3B3985BB45407F3F313CA244BABA1943FABDEBC0C8BC20B5CD4314C4C1C79344E0C2E9BF572925C2503D7EBEABC16C39F7C307C0DCC4DAB9003D54C3173A2C3ACBC46F40C8BC7FAAC442C63C5C427531E9C635397939724187B5B1BCFC4345BD8CBAA83D0038254525BF9DBAEF3BE5C05B28EC421C410DC4183F95C7ED4211390DA9B5BC394068C3C1404EB8A64162A9C140243E7D442446EA43053EC43F5F39D845D0C4A543D442692A92C309440D3EB7C493BD663BC1A38344853D0E46C635D9C230409239E13CB4C41A4571C04FC4B230E6B608C6583CCAC1722C0CC2F7B5ACC137C074B863BCABC396BC6F387C3C4FBC4CC0233C4AC2FBBF3FC46840C243B8BC663C24C071B9423CFF36A2C2294628AC4E3FBD36AEC6CDBC57B3D6BE3FC2EF39B53053C03042D5C068BC05B9863BD9406542A43927C146C4ACB6DC344C46DABA5D3D2D40B144213DCCB13BB71DBC9141F8BF6F4668C4DD410F4326BEEBB6E0C08144293F2EC588BBB6BF792F4F38E1BCFD4076455CACE8B4B63C9F401EC098C4F541A1C5AA40AAB5F1BD9FB5F23C75BE59BB3DB8923C4F4038BDBB3895BAC43A83C0CBC3B534A929743EE4C3CA2791C41EBFF544CCC3543F25BE21420E42263BD7386B3CC24314C4804158B9CE40DE3908C53EC2E4C4B93F014068BF05BFFBC1F14518C105C09148A3B5F24702BD1BC45F41A5366E3CF7BA5C38033F75B8E839FBBE3C3EA1C0304030BDA33DCF446B44383AC8413640CCBC4B398F398FBCF93CE0C4B2BE04446FBADCC1B83A1E425B3E55C299440AB53E4380C7A1BF8248BDC269B862BF68B122427C3CE4C41945B941B64168C4F0C40838CB2800BFDB34F2C41BBE8CBB18C416391533B641B54546C344BCEEC1B03BAE415142BFBC3E2885C4C4C2AB40B73A5BC2154517BD4FC86F3898B81FC47F4099410F4467BDAE4467340D46FCBB764036BFA0402A4063C5E63DC73B053E3C4503C6163CD0399EB8A0BD4F3D23C7CBBC1B3DC5C55EC3403F3FC2D6B88E443E447847B5C1E02BB841544176B1EAC0D2438FC00938FDC5E4C0C1B4F03E02C465C04342"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
  func.func private @expected() -> (tensor<20x20xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x45390F37DD352339C138F935E331163BF1385532B0335837AC361D346534B533C230F430A9379E38B637BF34223543371E3217381333E5355D34BB36F539B7328E32AA3029323433F834AD3BAC3353369B3504348B38A432EA34F73162388F36F3324C3845380232A434BB369C3B4433BD368633CB3AF430A03886381C34D539CF36A2325B36253818362939C931493520387D376034BC3B2D33443494324F35B8302C33AF38B23BCB36C734E932743404390634AD3B7434CC3539324531AA32DE35073590386831FF31CA323B339D3BD4329832D93510322536C737E13B35322F364F31C2393833CD347C38A7361232D031A3345D32F23A713952311D37F033793BBE33B4390434C834F2381337C832E636F538FD362637BA3450379233F2346A32A834BD32C9361137D634893832376B3959334231813B37357C390B31B736723A6E359A335A38F13AB634A43369340F37B438B5376634803376383F34643281390B3A33310D384A36CF3414327636BB3A5B3956370F34F33424314932E1331933CB357039633436324835C431B4370C35253B0339A73654349B317B3B073ACA368734DA342532CF3384318034CB39EA35CE399A369F35CE370C39E936B8346536D338223813389734B832173AA83BA035AD32C33B2A324D35E731B8323535CB35AF33BC33EC37C7380C37BD328E32163493386C346038DB319A33F1310B35EF342C355835CA335C314634EC344930CD399B308D3688322534833909370438FD385935F2385D385D35BE358634CD346A361B36FF3147324138F133C934B83698387D38EC369436F43180359C322F38E2331738B133AD358B332532FC39FF32BF30153551304833F7382F35CE3AAE33FD36F131D031FD3300344932EA312539B53B5A350B3AE831D135B2378A32AD387E3A00347A31FB323037D4339E3703348D33C336BE3B33324433803418388733D3317D366B30F538E23884329A340C34923243361632313A4F3176379F3442358634D134A531F1359237DE35BB3154315D376538E0381D365436DF30B9367A367231EE323D359A33C8382D326A32C2300034883BFE332A34CB3A5D34B5329034253957316134133A62359E32AA349733"> : tensor<20x20xf16>
    return %cst : tensor<20x20xf16>
  }
}
