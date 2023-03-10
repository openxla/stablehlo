// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<20x20xf32>, tensor<20x20xf32>)
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = call @igamma(%0#0, %0#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<20x20xf32>, tensor<20x20xf32>) {
    %0 = stablehlo.constant dense<"0x993B153EF7CCCBBFE8ACE6BF2532483F207206BF6E9C3CC0613592BF82CABEBFB059393FC01D56C006D93240B205E93E5DE00140238C623F464B8DBE06DCF2BF407891C02F0F8FBFFBBA334068B32A3E0C6A4B401C892AC04FDE7A3FC8C38B40376CB540C38EA2C0B75CE2BFC69E3F4051EABDBF4F4D63C0FD4D01BF509F8AC03B8C16BF09850B40B1055440FD8417C015FC17C0D72985C036F92840CEBA6CC0A04E353FFA218340520F1FBF374C69C0AACF66C01934B83FF15477403212EB3F83E935C0B9CFA73E40B2B13F06577340E5939EBF736FA8C0B85D0B401F32864032AA2840A5D65C3F060BFE3F8BC6AE401030953F4D478D3E03A0E5BFEBD765404277EA3FF30A69C006302E40C29A97C00B2225408CF00540A26644BFBBE6B4400BABC43FB5FC293E780FF0BFE6FA01BFA9F9A9BE16D5253E8F0C80BFD038A8C0E63D52BE98E5BA3F907EF3BF43DAB9C04312C4BE4A541FC095EA9E408FEE0F407A6DD5BF2CC63D3FEA7F05BFE5A72B405FAF214055769AC099E7704006A96FC037A5E9BF762E0ABFAEF91CBF4A6DC23F007129C0FB3FA1C023183FC0CCB29FC039653FBF539C15BF8D7A713EB070A3BFEFEF96BE282373C0385EC4BEF10EA1BF7BCDA4400FE3D33EEE74C5BD6314BABF487E7640037F11C09E7387C0A02BE73F432A53BF50CFBE40E118263FFEA702C06F5A1B40E8EF883F05C18340E917F33F1F89D73FFFA82F40E4A662BF621D833EDE8788C0708DF9C09B91FDBFC13415BE86A2C2BEEA9CC73F3FF536BD0280E63D3261DDBF596E12BF7880993F5F0228C09394DA3FD5E084C0605E0740A17BEDBF16A704BFE63ADC3BA4022340B50A0ABFA340623F1ACC2740DF249E3FFDB52DC00BAB6540B8FA104089FBD3BE442E663FDBAB733F942248BF58409C3F2D7C1CC0482C4D40FA228640BC4668BE12829F40256A69406849364005E5A3BF8AB591400D2485C0A748F63F2973D43FEB35513FA56EF4BF5F3CE3C07932E2BFAA4988C0B7DD31405D902CBC316D9FBF8C072740910FBCBF1093E8BFBA1C86C0307158C0C510FABFEC4678BE3D2338C0890BFC3EA64C9C3EA0E44B40F07582BF0A913E3D4B1402C0609E823F9F26F03F166409409A5AF73F85D0AD3FA6263F40FCC4C9BFAE8D4E3D06193F40A952A63D4FF16B3F7A1D05BF7FCD87C04662E43FF950DD3F16C623BFF962223ED60599C0CE46413FE7235840E48769C082493B402F62A4C07A10CB40E64546409C842F3E449145C065FC184095A567C0E970DE3CE85B553FCAE62BBF4C2973BF26BA583F570F5340BFCE7EC0C58C4E40BDCBAC40157343C00D0FEA3FAC3340C019DD8A40BDA004C00473AB3F5AF70B3FDEB14CBFDCA29E403CB9114027A1EA3FA4444CBFC52839C04B25F33E53B29DC05ACF4FC02AF057C0F0DED8BFC47403C051658C3F90230E40AD443DC0724BA4BFABC5BB3D1A8556C062BDD140814B1A40E50F8740AA84DD3FE289DC3F4B2176BF2774BC407AA84ABE1CA10D40BED8A3BED177994056E9FCBFF7F5E8409571F2BF02ED11C08772B1C08DAD4540998F15C00EAE8F3DA944F43C11741F3E9789563FF13E1540AB1A844054A5B1BE8CB84DC0DD657440790BD03E2D7AA8BF07BC0A3F823712BF57FFC2C0E21626C06F706B4013D87940982C2340F303823D6E96DDBFFA1480C08BAD2B40578484C0752BB93FB41C1C40FD7903401B866A402040D23F6BAFBA40B5285EC0ED00F0BAB11BCFC024A74940386D41BFA9DEB1BFCC6C0AC082AF71BEFFD12340821A44407C4471C0EDC69B3F2129853F090E8BBF742AD0BF73D314407F555240FABBA53EFBBED53FFFFBB53FB32BAF3ED4F3E7BF8C16F7BF08B8CD409A57C9BE4C1B07BF8E9B2040FF01A0BF393C8940395B03BC76763E3E89D42AC0441BD6BF101A64C036EF8E40E67584C0D919D840161A86BFFF870640D5CE4FC0661792BE4CF096401E50173F256BAA3F23AD034064A71240E619D7C00E82983F4EC24E4052753B40C2769B4044C93B3FD1CA51C097FCA4407B4A59C0396ECFC0BFBBDB3FF5201BC046D79C40EE2E34C0A066623F221B903FA7A6A8BFC61B14BFA91CC6BF8A9F0B40BB9EE8BFA1D71ABE8442693F0FEBDBC0A7218A408B25CC4041FF07C0806F39BEF7DC23401E8347C08F924EC0B2C9CE3FB469A83F0025823ED3582F40EA544EBE3906B53FD3C123BFE45D5A3F8D560640F90ACA3F0A2C30C0E1FD0DC0E14930401452EC40E3962F400E39B5BE"> : tensor<20x20xf32>
    %1 = stablehlo.constant dense<"0xACC5FA3FD46EA23FA8DA34C0630F81C094ACA4BF6527ADBF118CBD402E72F2BF2DF1D63EEBBD1B3FC96E464066A32D40EDDB99408EF826C0191777BFD74814409990A040D075C5C06A4BD8BF229D85403B1517C1611C8E3F9EEEB6C03C2E80C075FCFC3FBA2DE73F665A7A3F6F8AFFBFB39516C059BA864060371F3F49F45DC042D92FBFC2EC1240ECFCAAC05BB7553FEB6F11BF139D38C06A72E7BFAC8505BFA9AE0540022121C199B67CBF28936FC04ED62440AC6E87404ED88040610E023FD1C5FABF1DE395C01F40CF404F4234C083B9AF3F3C2930BF7C34683F57ABBE3F30B07BBF906B4CBEE375B63FB34D1AC00A312ABF93329A3EF550D23F56F984BD271037C086AF91C0E5960FBF68ABFD3F6869694051D4A1C039E1D0BF24402640532CECBF99251AC0B4640340783B9BC0BA05ECBEE1715C3FF72F02C05B0F8DBF1000D53E942AC23F722AB240EE25B73FEC2393C0D58478BF92380C3F5D48B6C00B3F5140CC921BC0818FFABF46FD30405CFE8ABF80B8A240D10679C0ED3E5DBFC75E55C0D2EE253F0251A0406694FDBE3BE8813FB9803FC028E569C03021DE3E6F3C97400C9532C06FC0FB3E76763E405DDD0C40A522A9C05EFA99402A4B42C02951084031BB27C0D56662C0482F8A4001F205C0FCF572BFCFA1FEBFAAB4A03F6D483D407A66E0BD63B7CE40AC69C13E3C773CC0EDE9C23E4324AD3E88E853BF699590C018130DC00A8A9FBF1A28C03ECFAF2CBFA39B8BC04D9FFCBF3259673F12BC1BBF2222ED3F5AD783BF247C9F3F7B875BC03EE4403FD70CE93FF21E8A4083F566BF21882F400286FB402E81A240FB73AEC04DBB54C06AB02B40DE3E0EBF297645C00B6C9FBEE174CC3FB70116BFEDB68AC073447A408BF7D1BF6AF9A73E576636C0EBE254BE4A9B09409D37C8C0BDE682C05E6531C061A339C0A5102140CC4C60BF2C0B07403FD1E7BE495BBBBE99F175BE29DFFBBE7269F6C0125129BE6D6F853FC36D7E3E1085BBC05A0DB2BFE72928409291DB3FC2228A40722DA3406AB2B33E84979CBEB00F4E40FBB71D40AB52BD3F268532C053DB454030A49DBFA7198F40D3A023C05A571BC09BBAAA3DDBC51DBFC087A6C0AF132F3FBFF8D2BF3A36AB407B0037C0D0110240AE0F663FF59F1D401BD965BF4C0EB1C01DFE733F299588401713D2C079BC9940617FA13E6689C2BCCB4F24C0A2DD873F032980BFCA0D4F40BCFC8DBF283CEA3FEB9AB6C0CC7EE73F2DA79F3DE526F340966E5D4048B7CB3DD5524CC0B10D7EBF9691C43E39649FBE181C973FB8C18ABEA36F204075D57F40A82A244058924AC0635BE1BF154FB6409FB36D409EFEB13F334596404C1313406B3D52C079885DBF88F3DC40897F9ABF4A45A6C0F3E016C0201D50BEA0477ABF6E2A1ABF9B9082C0B473A840474BDC406FE934400F318E4020DF033F858391BD353295C0AC3020C029768C408DFFC7C0E227D23F443012BED566B94085F62ABF3B47603FD1F051C08E7A24C011A7C0BF1E731CBF320ED43F58653D405C3A8B408FA2B9C012EC17400E18BCC0CC06923F85FF7CBFBECB0DC08FD465C0BBEE2E400B615AC0399441C04FE649C0369E0CC038161D40F4564BC0A755FCC0FA5F8B40AB325FBFAC8F5ABF510D9B40DF76B5C08055D4C0B6073F3FEEB2AD3F897457C0A0C5C7BE46A4BBC04B337FC096A959406F3CA5C00893AE409E9686401D859440F26ACB3F56C76F40AE0E903F165CB8C0D9992F40FE21EC3FE057A4BFEC96F33FDBF05440452B7B3F511C61407EEABA3F03E54AC04AC46AC09CC800C04673AC405CFFD4BF7B031DC029813040D45427C005053CC0DC8E16C0729C2440F2812D40784ADAC00EED333F7306823FCFBEA43D9B09B93F8800B4BF106DAFBE66A5A33F7C60CAC084C838401CF03840DCA281C02AF06C3F745700BF26A480BE3B53AB3F1C95523F8AB0A640F1E4243F41958D40A296543F9245374012D81EBF10EAF7BFAE10BFC035A52C3FAC140DC0887670C03687C0BF45F56140D7BA4640671270C00DCABB3F8A1B604046263FC062ADF93F270DD8C083829CC014EC4C405B5AA8C08C9895BF8FFC6CC06CD86D3F67C095C05E6122C072ED0BC03272DB3E93FCEA3F3B86A9BEB19F134010203040ED529ABE6BF6693FA644313E19D6CA4021C600C094EFC1BF86F0894062EF533F4EBA103FBC14BBBF323C65C01CD911BFABCD93402C85C73EDA178B3FCA7B5D3DB77ABA3E29FAC240E0B444BFC4EEE63F"> : tensor<20x20xf32>
    return %0, %1 : tensor<20x20xf32>, tensor<20x20xf32>
  }
  func.func private @expected() -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<"0x829E7D3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F65A2FC3E0000C07F6381253FC1A07B3F1760733F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FF4D07F3F0000C07F0000C07F0000C07F0000C07F2AA5C03C0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FE78D1E3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FA6F96D3F0000C07F0000C07F0000C07F0000C07F275B773F4E61193FFDAFF73D0000C07F0000C07F69087F3F0000C07F0000C07F0000C07F96CB3F3EF614513D0000C07F0000C07F88EBD73E0000C07F0000C07F4AB33F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F853B493F0000C07F0000C07FD9BE8B3D0000C07F0000C07F0000C07F0000C07F0000C07F1ADB723F0000C07F0000C07F0000C07F7D90203F0000C07F0000C07F0000C07F0000C07FEE1394390000C07F0000C07F0000C07F0000C07FC4C0193F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FD07B5A3F0000C07F0000C07F0000C07F0000C07F0000C07F236C643D0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F85A4D63E0000C07F0000C07F5BDB7F3F0000C07F0000C07F76F0913E4E99A0390000C07F0000C07F0000C07F0000C07FEFD34C3F0000C07F0000C07F0000C07F0000C07F0000C07F78E32F3F0000C07FF1307B3F0000C07F0000C07F57F4483F0000C07F0000C07F0000C07F9CEE7E3F0000C07F0000C07F0000C07F99821D3F0000C07F0000C07F0000C07FBF6A393F0000C07F0000C07F362E5E3F0000C07F8223A83E0000C07F0000C07FE938563F0000C07F0000C07F0000C07F0000C07F4783E63D0000C07FE2C4C73E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FFB2B0B3F0000C07F0000C07F80D26B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F79B07F3F0000C07F0000C07F7301693F0000C07F0000C07FD41A363E0000C07FFA01793F0000C07F689DAB3E0000C07FD2A17F3F0000C07F0000C07F3031263F0000C07F0000C07F0009773F58B6903D0000C07F0000C07F0000C07F0000C07F7CAD0A3F0000C07F1976963E0000C07F35FED93BF8B25638F8FE7F3F0000C07F8C74A63A0000C07F0000C07F2129CF3E0000C07F0000C07F0000C07FB505C63E0000C07F20F0D53E0000C07F0000C07FDC957B3F0000C07FFB030C3D0000C07FEF26553F0000C07F0000C07F44B0523F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F53487C3FA4D8853D0000C07F0000C07F0000C07F0000C07F0000C07F8533BC3E0000C07FD28C7C3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F681A0E3D0000C07F68DBFF3D0000C07F0000C07F0000C07F6A76C63D0000C07F0000C07F0000C07F9D0B7F3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FAAE70B3CCF557A3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FFF57733F45C36A3FAF5F3D3F8D6A163FF884493E0000C07F0000C07F0000C07FB2677F3E0000C07F0000C07F0000C07F0000C07F484E453F272B2E3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FE345E73E0000C07F0000C07F0000C07F83A57C3F0000C07F0000C07F29B611380000C07F0000C07F54D38F3E0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F46BDDD3B0000C07F0000C07F0000C07F2CFB333E0000C07F0000C07FA4C8013FE575423F5090663F0000C07F0000C07F0000C07FD1F9CD3E0000C07F0000C07F0000C07FC9F47B3F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F8FD1773F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07FC74DC93E0000C07F0000C07FDB5EA23C0000C07F0000C07FDACDF13D0000C07F0000C07F0000C07F0000C07F94B77F3F1C909A3D0000C07F0000C07F0000C07F0000C07F7DE16F3FA334023E0000C07F0000C07F231C2F3CB7EDB33E0000C07F0000C07F"> : tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igamma(%arg0: tensor<20x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
    %0 = call @xla_fallback_igamma(%arg0, %arg1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    return %0 : tensor<20x20xf32>
  }
  func.func private @igammac_body.199(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
    %0 = stablehlo.get_tuple_element %arg0[7] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %4 = stablehlo.add %1, %3 : tensor<20x20xf32>
    %5 = stablehlo.multiply %0, %4 : tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[9] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %7 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %10 = stablehlo.add %7, %9 : tensor<20x20xf32>
    %11 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<f32>
    %12 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %13 = stablehlo.add %11, %12 : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %15 = stablehlo.multiply %10, %14 : tensor<20x20xf32>
    %16 = stablehlo.multiply %6, %15 : tensor<20x20xf32>
    %17 = stablehlo.subtract %5, %16 : tensor<20x20xf32>
    %18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %20 = stablehlo.compare  NE, %17, %19 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %21 = stablehlo.get_tuple_element %arg0[12] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %22 = stablehlo.multiply %21, %4 : tensor<20x20xf32>
    %23 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %24 = stablehlo.subtract %22, %23 : tensor<20x20xf32>
    %25 = stablehlo.get_tuple_element %arg0[10] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %26 = stablehlo.multiply %25, %15 : tensor<20x20xf32>
    %27 = stablehlo.subtract %24, %26 : tensor<20x20xf32>
    %28 = stablehlo.get_tuple_element %arg0[8] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %29 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %30 = stablehlo.multiply %28, %29 : tensor<20x20xf32>
    %31 = stablehlo.add %27, %30 : tensor<20x20xf32>
    %32 = stablehlo.multiply %23, %4 : tensor<20x20xf32>
    %33 = stablehlo.multiply %28, %15 : tensor<20x20xf32>
    %34 = stablehlo.subtract %32, %33 : tensor<20x20xf32>
    %35 = stablehlo.divide %34, %17 : tensor<20x20xf32>
    %36 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %37 = stablehlo.select %20, %35, %36 : tensor<20x20xi1>, tensor<20x20xf32>
    %38 = stablehlo.get_tuple_element %arg0[13] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %39 = stablehlo.multiply %38, %4 : tensor<20x20xf32>
    %40 = stablehlo.subtract %39, %0 : tensor<20x20xf32>
    %41 = stablehlo.get_tuple_element %arg0[11] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %42 = stablehlo.multiply %41, %15 : tensor<20x20xf32>
    %43 = stablehlo.subtract %40, %42 : tensor<20x20xf32>
    %44 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %45 = stablehlo.multiply %6, %44 : tensor<20x20xf32>
    %46 = stablehlo.add %43, %45 : tensor<20x20xf32>
    %47 = stablehlo.multiply %37, %46 : tensor<20x20xf32>
    %48 = stablehlo.subtract %31, %47 : tensor<20x20xf32>
    %49 = stablehlo.divide %48, %17 : tensor<20x20xf32>
    %50 = stablehlo.get_tuple_element %arg0[14] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %51 = stablehlo.select %20, %49, %50 : tensor<20x20xi1>, tensor<20x20xf32>
    %52 = stablehlo.subtract %51, %50 : tensor<20x20xf32>
    %53 = stablehlo.abs %52 : tensor<20x20xf32>
    %54 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %55 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %56 = stablehlo.select %20, %53, %55 : tensor<20x20xi1>, tensor<20x20xf32>
    %57 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %58 = stablehlo.subtract %36, %35 : tensor<20x20xf32>
    %59 = stablehlo.divide %58, %35 : tensor<20x20xf32>
    %60 = stablehlo.abs %59 : tensor<20x20xf32>
    %61 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %62 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %63 = stablehlo.select %20, %60, %62 : tensor<20x20xi1>, tensor<20x20xf32>
    %64 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %65 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %66 = stablehlo.compare  GT, %63, %65 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %67 = stablehlo.and %57, %66 : tensor<20x20xi1>
    %68 = stablehlo.select %57, %37, %36 : tensor<20x20xi1>, tensor<20x20xf32>
    %69 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %70 = stablehlo.select %57, %63, %69 : tensor<20x20xi1>, tensor<20x20xf32>
    %71 = stablehlo.select %57, %10, %7 : tensor<20x20xi1>, tensor<20x20xf32>
    %72 = stablehlo.select %57, %4, %1 : tensor<20x20xi1>, tensor<20x20xf32>
    %73 = stablehlo.abs %34 : tensor<20x20xf32>
    %74 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %75 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %76 = stablehlo.constant dense<0x4B000000> : tensor<f32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %78 = stablehlo.compare  GT, %73, %77 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %79 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %80 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %81 = stablehlo.multiply %34, %80 : tensor<20x20xf32>
    %82 = stablehlo.select %78, %81, %34 : tensor<20x20xi1>, tensor<20x20xf32>
    %83 = stablehlo.select %57, %82, %23 : tensor<20x20xi1>, tensor<20x20xf32>
    %84 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %86 = stablehlo.multiply %17, %85 : tensor<20x20xf32>
    %87 = stablehlo.select %78, %86, %17 : tensor<20x20xi1>, tensor<20x20xf32>
    %88 = stablehlo.select %57, %87, %0 : tensor<20x20xi1>, tensor<20x20xf32>
    %89 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %90 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %91 = stablehlo.multiply %23, %90 : tensor<20x20xf32>
    %92 = stablehlo.select %78, %91, %23 : tensor<20x20xi1>, tensor<20x20xf32>
    %93 = stablehlo.select %57, %92, %28 : tensor<20x20xi1>, tensor<20x20xf32>
    %94 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %95 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %96 = stablehlo.multiply %0, %95 : tensor<20x20xf32>
    %97 = stablehlo.select %78, %96, %0 : tensor<20x20xi1>, tensor<20x20xf32>
    %98 = stablehlo.select %57, %97, %6 : tensor<20x20xi1>, tensor<20x20xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %100 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %101 = stablehlo.multiply %21, %100 : tensor<20x20xf32>
    %102 = stablehlo.select %78, %101, %21 : tensor<20x20xi1>, tensor<20x20xf32>
    %103 = stablehlo.select %57, %102, %25 : tensor<20x20xi1>, tensor<20x20xf32>
    %104 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %105 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %106 = stablehlo.multiply %38, %105 : tensor<20x20xf32>
    %107 = stablehlo.select %78, %106, %38 : tensor<20x20xi1>, tensor<20x20xf32>
    %108 = stablehlo.select %57, %107, %41 : tensor<20x20xi1>, tensor<20x20xf32>
    %109 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %110 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %111 = stablehlo.multiply %31, %110 : tensor<20x20xf32>
    %112 = stablehlo.select %78, %111, %31 : tensor<20x20xi1>, tensor<20x20xf32>
    %113 = stablehlo.select %57, %112, %21 : tensor<20x20xi1>, tensor<20x20xf32>
    %114 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %115 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %116 = stablehlo.multiply %46, %115 : tensor<20x20xf32>
    %117 = stablehlo.select %78, %116, %46 : tensor<20x20xi1>, tensor<20x20xf32>
    %118 = stablehlo.select %57, %117, %38 : tensor<20x20xi1>, tensor<20x20xf32>
    %119 = stablehlo.select %57, %51, %50 : tensor<20x20xi1>, tensor<20x20xf32>
    %120 = stablehlo.tuple %67, %68, %70, %71, %72, %13, %83, %88, %93, %98, %103, %108, %113, %118, %119 {xla_shape = "(pred[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=5*/f32[], f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=10*/f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0})"} : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
    return %120 : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
  }
  func.func private @or.322(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igammac_condition.326(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
    %0 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %5 = stablehlo.get_tuple_element %arg0[7] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[8] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %7 = stablehlo.get_tuple_element %arg0[9] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %8 = stablehlo.get_tuple_element %arg0[10] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %9 = stablehlo.get_tuple_element %arg0[11] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %10 = stablehlo.get_tuple_element %arg0[12] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %11 = stablehlo.get_tuple_element %arg0[13] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %12 = stablehlo.get_tuple_element %arg0[14] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %13 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<f32>
    %14 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
    %15 = stablehlo.compare  LT, %13, %14 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %16 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %17 = stablehlo.constant dense<false> : tensor<i1>
    %18 = stablehlo.reduce(%16 init: %17) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
     reducer(%arg1: tensor<i1>, %arg2: tensor<i1>)  {
      %20 = stablehlo.or %arg1, %arg2 : tensor<i1>
      stablehlo.return %20 : tensor<i1>
    }
    %19 = stablehlo.and %15, %18 : tensor<i1>
    return %19 : tensor<i1>
  }
  func.func private @igamma_body.378(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>> {
    %0 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %5 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %6 = stablehlo.add %3, %5 : tensor<20x20xf32>
    %7 = stablehlo.divide %2, %6 : tensor<20x20xf32>
    %8 = stablehlo.multiply %1, %7 : tensor<20x20xf32>
    %9 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %10 = stablehlo.add %9, %8 : tensor<20x20xf32>
    %11 = stablehlo.divide %8, %10 : tensor<20x20xf32>
    %12 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %13 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
    %14 = stablehlo.compare  GT, %11, %13 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %15 = stablehlo.and %0, %14 : tensor<20x20xi1>
    %16 = stablehlo.select %0, %6, %3 : tensor<20x20xi1>, tensor<20x20xf32>
    %17 = stablehlo.select %0, %8, %1 : tensor<20x20xi1>, tensor<20x20xf32>
    %18 = stablehlo.select %0, %10, %9 : tensor<20x20xi1>, tensor<20x20xf32>
    %19 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %20 = stablehlo.divide %2, %6 : tensor<20x20xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<20x20xf32>
    %22 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %23 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
    %24 = stablehlo.multiply %23, %1 : tensor<20x20xf32>
    %25 = stablehlo.multiply %24, %2 : tensor<20x20xf32>
    %26 = stablehlo.multiply %6, %6 : tensor<20x20xf32>
    %27 = stablehlo.divide %25, %26 : tensor<20x20xf32>
    %28 = stablehlo.add %21, %27 : tensor<20x20xf32>
    %29 = stablehlo.select %0, %28, %19 : tensor<20x20xi1>, tensor<20x20xf32>
    %30 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %31 = stablehlo.add %30, %28 : tensor<20x20xf32>
    %32 = stablehlo.select %0, %31, %30 : tensor<20x20xi1>, tensor<20x20xf32>
    %33 = stablehlo.tuple %15, %16, %17, %18, %2, %29, %32 {xla_shape = "(pred[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, f32[20,20]{1,0}, /*index=5*/f32[20,20]{1,0}, f32[20,20]{1,0})"} : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
    return %33 : tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>
  }
  func.func private @or.415(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @igamma_condition.419(%arg0: tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<i1> {
    %0 = stablehlo.get_tuple_element %arg0[1] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %1 = stablehlo.get_tuple_element %arg0[2] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %2 = stablehlo.get_tuple_element %arg0[3] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %3 = stablehlo.get_tuple_element %arg0[4] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %4 = stablehlo.get_tuple_element %arg0[5] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %5 = stablehlo.get_tuple_element %arg0[6] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xf32>
    %6 = stablehlo.get_tuple_element %arg0[0] : (tuple<tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>>) -> tensor<20x20xi1>
    %7 = stablehlo.constant dense<false> : tensor<i1>
    %8 = stablehlo.reduce(%6 init: %7) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
     reducer(%arg1: tensor<i1>, %arg2: tensor<i1>)  {
      %9 = stablehlo.or %arg1, %arg2 : tensor<i1>
      stablehlo.return %9 : tensor<i1>
    }
    return %8 : tensor<i1>
  }
  func.func private @xla_fallback_igamma(%arg0: tensor<20x20xf32>, %arg1: tensor<20x20xf32>) -> tensor<20x20xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %2 = stablehlo.compare  EQ, %arg1, %1 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %5 = stablehlo.compare  LT, %arg1, %4 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %8 = stablehlo.compare  LE, %arg0, %7 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %9 = stablehlo.or %5, %8 : tensor<20x20xi1>
    %10 = stablehlo.or %2, %9 : tensor<20x20xi1>
    %11 = stablehlo.log %arg1 : tensor<20x20xf32>
    %12 = stablehlo.multiply %arg0, %11 : tensor<20x20xf32>
    %13 = stablehlo.subtract %12, %arg1 : tensor<20x20xf32>
    %14 = stablehlo.abs %arg0 : tensor<20x20xf32>
    %15 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %16 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %17 = stablehlo.compare  EQ, %14, %16 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %19 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %20 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %21 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %22 = stablehlo.compare  LT, %arg0, %21 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %23 = stablehlo.constant dense<3.14159274> : tensor<f32>
    %24 = stablehlo.constant dense<3.14159274> : tensor<20x20xf32>
    %25 = stablehlo.abs %arg0 : tensor<20x20xf32>
    %26 = stablehlo.floor %25 : tensor<20x20xf32>
    %27 = stablehlo.subtract %25, %26 : tensor<20x20xf32>
    %28 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %29 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %30 = stablehlo.compare  GT, %27, %29 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %31 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %32 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %33 = stablehlo.subtract %32, %27 : tensor<20x20xf32>
    %34 = stablehlo.select %30, %33, %27 : tensor<20x20xi1>, tensor<20x20xf32>
    %35 = stablehlo.multiply %24, %34 : tensor<20x20xf32>
    %36 = stablehlo.sine %35 : tensor<20x20xf32>
    %37 = stablehlo.log %36 : tensor<20x20xf32>
    %38 = stablehlo.is_finite %37 : (tensor<20x20xf32>) -> tensor<20x20xi1>
    %39 = stablehlo.constant dense<1.14472985> : tensor<f32>
    %40 = stablehlo.constant dense<1.14472985> : tensor<20x20xf32>
    %41 = stablehlo.subtract %40, %37 : tensor<20x20xf32>
    %42 = stablehlo.constant dense<0.918938517> : tensor<f32>
    %43 = stablehlo.constant dense<0.918938517> : tensor<20x20xf32>
    %44 = stablehlo.negate %arg0 : tensor<20x20xf32>
    %45 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %46 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %47 = stablehlo.subtract %arg0, %46 : tensor<20x20xf32>
    %48 = stablehlo.select %22, %44, %47 : tensor<20x20xi1>, tensor<20x20xf32>
    %49 = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf32>
    %50 = stablehlo.add %48, %49 : tensor<20x20xf32>
    %51 = stablehlo.constant dense<7.500000e+00> : tensor<f32>
    %52 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %53 = stablehlo.add %52, %48 : tensor<20x20xf32>
    %54 = stablehlo.constant dense<2.01490307> : tensor<f32>
    %55 = stablehlo.constant dense<2.01490307> : tensor<20x20xf32>
    %56 = stablehlo.constant dense<7.500000e+00> : tensor<20x20xf32>
    %57 = stablehlo.divide %48, %56 : tensor<20x20xf32>
    %58 = stablehlo.log_plus_one %57 : tensor<20x20xf32>
    %59 = stablehlo.add %55, %58 : tensor<20x20xf32>
    %60 = stablehlo.divide %53, %59 : tensor<20x20xf32>
    %61 = stablehlo.subtract %50, %60 : tensor<20x20xf32>
    %62 = stablehlo.multiply %61, %59 : tensor<20x20xf32>
    %63 = stablehlo.add %43, %62 : tensor<20x20xf32>
    %64 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %65 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %66 = stablehlo.constant dense<676.520386> : tensor<f32>
    %67 = stablehlo.constant dense<676.520386> : tensor<20x20xf32>
    %68 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %69 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %70 = stablehlo.add %48, %69 : tensor<20x20xf32>
    %71 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %72 = stablehlo.add %70, %71 : tensor<20x20xf32>
    %73 = stablehlo.divide %67, %72 : tensor<20x20xf32>
    %74 = stablehlo.add %65, %73 : tensor<20x20xf32>
    %75 = stablehlo.constant dense<-1259.13916> : tensor<f32>
    %76 = stablehlo.constant dense<-1259.13916> : tensor<20x20xf32>
    %77 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %78 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %79 = stablehlo.add %48, %78 : tensor<20x20xf32>
    %80 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %81 = stablehlo.add %79, %80 : tensor<20x20xf32>
    %82 = stablehlo.divide %76, %81 : tensor<20x20xf32>
    %83 = stablehlo.add %74, %82 : tensor<20x20xf32>
    %84 = stablehlo.constant dense<771.323425> : tensor<f32>
    %85 = stablehlo.constant dense<771.323425> : tensor<20x20xf32>
    %86 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %87 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
    %88 = stablehlo.add %48, %87 : tensor<20x20xf32>
    %89 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %90 = stablehlo.add %88, %89 : tensor<20x20xf32>
    %91 = stablehlo.divide %85, %90 : tensor<20x20xf32>
    %92 = stablehlo.add %83, %91 : tensor<20x20xf32>
    %93 = stablehlo.constant dense<-176.615036> : tensor<f32>
    %94 = stablehlo.constant dense<-176.615036> : tensor<20x20xf32>
    %95 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %96 = stablehlo.constant dense<3.000000e+00> : tensor<20x20xf32>
    %97 = stablehlo.add %48, %96 : tensor<20x20xf32>
    %98 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %99 = stablehlo.add %97, %98 : tensor<20x20xf32>
    %100 = stablehlo.divide %94, %99 : tensor<20x20xf32>
    %101 = stablehlo.add %92, %100 : tensor<20x20xf32>
    %102 = stablehlo.constant dense<12.5073433> : tensor<f32>
    %103 = stablehlo.constant dense<12.5073433> : tensor<20x20xf32>
    %104 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %105 = stablehlo.constant dense<4.000000e+00> : tensor<20x20xf32>
    %106 = stablehlo.add %48, %105 : tensor<20x20xf32>
    %107 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %108 = stablehlo.add %106, %107 : tensor<20x20xf32>
    %109 = stablehlo.divide %103, %108 : tensor<20x20xf32>
    %110 = stablehlo.add %101, %109 : tensor<20x20xf32>
    %111 = stablehlo.constant dense<-0.138571098> : tensor<f32>
    %112 = stablehlo.constant dense<-0.138571098> : tensor<20x20xf32>
    %113 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %114 = stablehlo.constant dense<5.000000e+00> : tensor<20x20xf32>
    %115 = stablehlo.add %48, %114 : tensor<20x20xf32>
    %116 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %117 = stablehlo.add %115, %116 : tensor<20x20xf32>
    %118 = stablehlo.divide %112, %117 : tensor<20x20xf32>
    %119 = stablehlo.add %110, %118 : tensor<20x20xf32>
    %120 = stablehlo.constant dense<9.98436917E-6> : tensor<f32>
    %121 = stablehlo.constant dense<9.98436917E-6> : tensor<20x20xf32>
    %122 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %123 = stablehlo.constant dense<6.000000e+00> : tensor<20x20xf32>
    %124 = stablehlo.add %48, %123 : tensor<20x20xf32>
    %125 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %126 = stablehlo.add %124, %125 : tensor<20x20xf32>
    %127 = stablehlo.divide %121, %126 : tensor<20x20xf32>
    %128 = stablehlo.add %119, %127 : tensor<20x20xf32>
    %129 = stablehlo.constant dense<1.50563267E-7> : tensor<f32>
    %130 = stablehlo.constant dense<1.50563267E-7> : tensor<20x20xf32>
    %131 = stablehlo.constant dense<7.000000e+00> : tensor<f32>
    %132 = stablehlo.constant dense<7.000000e+00> : tensor<20x20xf32>
    %133 = stablehlo.add %48, %132 : tensor<20x20xf32>
    %134 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %135 = stablehlo.add %133, %134 : tensor<20x20xf32>
    %136 = stablehlo.divide %130, %135 : tensor<20x20xf32>
    %137 = stablehlo.add %128, %136 : tensor<20x20xf32>
    %138 = stablehlo.log %137 : tensor<20x20xf32>
    %139 = stablehlo.add %63, %138 : tensor<20x20xf32>
    %140 = stablehlo.subtract %41, %139 : tensor<20x20xf32>
    %141 = stablehlo.negate %37 : tensor<20x20xf32>
    %142 = stablehlo.select %38, %140, %141 : tensor<20x20xi1>, tensor<20x20xf32>
    %143 = stablehlo.select %22, %142, %139 : tensor<20x20xi1>, tensor<20x20xf32>
    %144 = stablehlo.select %17, %19, %143 : tensor<20x20xi1>, tensor<20x20xf32>
    %145 = stablehlo.subtract %13, %144 : tensor<20x20xf32>
    %146 = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %147 = stablehlo.constant dense<88.7228394> : tensor<f32>
    %148 = stablehlo.negate %147 : tensor<f32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
    %150 = stablehlo.compare  LT, %145, %149 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %151 = stablehlo.or %10, %150 : tensor<20x20xi1>
    %152 = stablehlo.compare  NE, %arg0, %arg0 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %153 = stablehlo.compare  NE, %arg1, %arg1 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %154 = stablehlo.or %152, %153 : tensor<20x20xi1>
    %155 = stablehlo.or %151, %154 : tensor<20x20xi1>
    %156 = stablehlo.not %155 : tensor<20x20xi1>
    %157 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %158 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %159 = stablehlo.compare  GT, %arg1, %158 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %160 = stablehlo.compare  GT, %arg1, %arg0 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %161 = stablehlo.and %159, %160 : tensor<20x20xi1>
    %162 = stablehlo.and %156, %161 : tensor<20x20xi1>
    %163 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %164 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %165 = stablehlo.add %arg1, %164 : tensor<20x20xf32>
    %166 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %167 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %168 = stablehlo.subtract %167, %arg0 : tensor<20x20xf32>
    %169 = stablehlo.add %arg1, %168 : tensor<20x20xf32>
    %170 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %171 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %172 = stablehlo.add %169, %171 : tensor<20x20xf32>
    %173 = stablehlo.multiply %172, %arg1 : tensor<20x20xf32>
    %174 = stablehlo.divide %165, %173 : tensor<20x20xf32>
    %175 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %176 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %178 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %179 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %180 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %181 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %183 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %184 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %185 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %186 = stablehlo.negate %arg1 : tensor<20x20xf32>
    %187 = stablehlo.multiply %174, %186 : tensor<20x20xf32>
    %188 = stablehlo.subtract %185, %187 : tensor<20x20xf32>
    %189 = stablehlo.divide %188, %173 : tensor<20x20xf32>
    %190:15 = stablehlo.while(%iterArg = %162, %iterArg_0 = %174, %iterArg_1 = %176, %iterArg_2 = %168, %iterArg_3 = %172, %iterArg_4 = %177, %iterArg_5 = %165, %iterArg_6 = %173, %iterArg_7 = %179, %iterArg_8 = %arg1, %iterArg_9 = %181, %iterArg_10 = %183, %iterArg_11 = %185, %iterArg_12 = %186, %iterArg_13 = %189) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %223 = stablehlo.constant dense<2.000000e+03> : tensor<f32>
      %224 = stablehlo.compare  LT, %iterArg_4, %223 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %225 = stablehlo.constant dense<false> : tensor<i1>
      %226 = stablehlo.reduce(%iterArg init: %225) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %228 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %228 : tensor<i1>
      }
      %227 = stablehlo.and %224, %226 : tensor<i1>
      stablehlo.return %227 : tensor<i1>
    } do {
      %223 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %224 = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf32>
      %225 = stablehlo.add %iterArg_3, %224 : tensor<20x20xf32>
      %226 = stablehlo.multiply %iterArg_6, %225 : tensor<20x20xf32>
      %227 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %228 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %229 = stablehlo.add %iterArg_2, %228 : tensor<20x20xf32>
      %230 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %231 = stablehlo.add %iterArg_4, %230 : tensor<f32>
      %232 = stablehlo.broadcast_in_dim %231, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %233 = stablehlo.multiply %229, %232 : tensor<20x20xf32>
      %234 = stablehlo.multiply %iterArg_8, %233 : tensor<20x20xf32>
      %235 = stablehlo.subtract %226, %234 : tensor<20x20xf32>
      %236 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %237 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
      %238 = stablehlo.compare  NE, %235, %237 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %239 = stablehlo.multiply %iterArg_11, %225 : tensor<20x20xf32>
      %240 = stablehlo.subtract %239, %iterArg_5 : tensor<20x20xf32>
      %241 = stablehlo.multiply %iterArg_9, %233 : tensor<20x20xf32>
      %242 = stablehlo.subtract %240, %241 : tensor<20x20xf32>
      %243 = stablehlo.broadcast_in_dim %231, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %244 = stablehlo.multiply %iterArg_7, %243 : tensor<20x20xf32>
      %245 = stablehlo.add %242, %244 : tensor<20x20xf32>
      %246 = stablehlo.multiply %iterArg_5, %225 : tensor<20x20xf32>
      %247 = stablehlo.multiply %iterArg_7, %233 : tensor<20x20xf32>
      %248 = stablehlo.subtract %246, %247 : tensor<20x20xf32>
      %249 = stablehlo.divide %248, %235 : tensor<20x20xf32>
      %250 = stablehlo.select %238, %249, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %251 = stablehlo.multiply %iterArg_12, %225 : tensor<20x20xf32>
      %252 = stablehlo.subtract %251, %iterArg_6 : tensor<20x20xf32>
      %253 = stablehlo.multiply %iterArg_10, %233 : tensor<20x20xf32>
      %254 = stablehlo.subtract %252, %253 : tensor<20x20xf32>
      %255 = stablehlo.broadcast_in_dim %231, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %256 = stablehlo.multiply %iterArg_8, %255 : tensor<20x20xf32>
      %257 = stablehlo.add %254, %256 : tensor<20x20xf32>
      %258 = stablehlo.multiply %250, %257 : tensor<20x20xf32>
      %259 = stablehlo.subtract %245, %258 : tensor<20x20xf32>
      %260 = stablehlo.divide %259, %235 : tensor<20x20xf32>
      %261 = stablehlo.select %238, %260, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      %262 = stablehlo.subtract %261, %iterArg_13 : tensor<20x20xf32>
      %263 = stablehlo.abs %262 : tensor<20x20xf32>
      %264 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %265 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %266 = stablehlo.select %238, %263, %265 : tensor<20x20xi1>, tensor<20x20xf32>
      %267 = stablehlo.subtract %iterArg_0, %249 : tensor<20x20xf32>
      %268 = stablehlo.divide %267, %249 : tensor<20x20xf32>
      %269 = stablehlo.abs %268 : tensor<20x20xf32>
      %270 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %271 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %272 = stablehlo.select %238, %269, %271 : tensor<20x20xi1>, tensor<20x20xf32>
      %273 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %274 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %275 = stablehlo.compare  GT, %272, %274 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %276 = stablehlo.and %iterArg, %275 : tensor<20x20xi1>
      %277 = stablehlo.select %iterArg, %250, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %278 = stablehlo.select %iterArg, %272, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %279 = stablehlo.select %iterArg, %229, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %280 = stablehlo.select %iterArg, %225, %iterArg_3 : tensor<20x20xi1>, tensor<20x20xf32>
      %281 = stablehlo.abs %248 : tensor<20x20xf32>
      %282 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %283 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %284 = stablehlo.constant dense<0x4B000000> : tensor<f32>
      %285 = stablehlo.broadcast_in_dim %284, dims = [] : (tensor<f32>) -> tensor<20x20xf32>
      %286 = stablehlo.compare  GT, %281, %285 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %287 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %288 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %289 = stablehlo.multiply %248, %288 : tensor<20x20xf32>
      %290 = stablehlo.select %286, %289, %248 : tensor<20x20xi1>, tensor<20x20xf32>
      %291 = stablehlo.select %iterArg, %290, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %292 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %293 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %294 = stablehlo.multiply %235, %293 : tensor<20x20xf32>
      %295 = stablehlo.select %286, %294, %235 : tensor<20x20xi1>, tensor<20x20xf32>
      %296 = stablehlo.select %iterArg, %295, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %297 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %298 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %299 = stablehlo.multiply %iterArg_5, %298 : tensor<20x20xf32>
      %300 = stablehlo.select %286, %299, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      %301 = stablehlo.select %iterArg, %300, %iterArg_7 : tensor<20x20xi1>, tensor<20x20xf32>
      %302 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %303 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %304 = stablehlo.multiply %iterArg_6, %303 : tensor<20x20xf32>
      %305 = stablehlo.select %286, %304, %iterArg_6 : tensor<20x20xi1>, tensor<20x20xf32>
      %306 = stablehlo.select %iterArg, %305, %iterArg_8 : tensor<20x20xi1>, tensor<20x20xf32>
      %307 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %308 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %309 = stablehlo.multiply %iterArg_11, %308 : tensor<20x20xf32>
      %310 = stablehlo.select %286, %309, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %311 = stablehlo.select %iterArg, %310, %iterArg_9 : tensor<20x20xi1>, tensor<20x20xf32>
      %312 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %313 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %314 = stablehlo.multiply %iterArg_12, %313 : tensor<20x20xf32>
      %315 = stablehlo.select %286, %314, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %316 = stablehlo.select %iterArg, %315, %iterArg_10 : tensor<20x20xi1>, tensor<20x20xf32>
      %317 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %318 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %319 = stablehlo.multiply %245, %318 : tensor<20x20xf32>
      %320 = stablehlo.select %286, %319, %245 : tensor<20x20xi1>, tensor<20x20xf32>
      %321 = stablehlo.select %iterArg, %320, %iterArg_11 : tensor<20x20xi1>, tensor<20x20xf32>
      %322 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %323 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %324 = stablehlo.multiply %257, %323 : tensor<20x20xf32>
      %325 = stablehlo.select %286, %324, %257 : tensor<20x20xi1>, tensor<20x20xf32>
      %326 = stablehlo.select %iterArg, %325, %iterArg_12 : tensor<20x20xi1>, tensor<20x20xf32>
      %327 = stablehlo.select %iterArg, %261, %iterArg_13 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %276, %277, %278, %279, %280, %231, %291, %296, %301, %306, %311, %316, %321, %326, %327 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %191 = stablehlo.not %161 : tensor<20x20xi1>
    %192 = stablehlo.and %156, %191 : tensor<20x20xi1>
    %193 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %194 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %195 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %196 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %197 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %198 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %199 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %200 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %201:7 = stablehlo.while(%iterArg = %192, %iterArg_0 = %arg0, %iterArg_1 = %194, %iterArg_2 = %196, %iterArg_3 = %arg1, %iterArg_4 = %198, %iterArg_5 = %200) : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
     cond {
      %223 = stablehlo.constant dense<false> : tensor<i1>
      %224 = stablehlo.reduce(%iterArg init: %223) across dimensions = [0, 1] : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
       reducer(%arg2: tensor<i1>, %arg3: tensor<i1>)  {
        %225 = stablehlo.or %arg2, %arg3 : tensor<i1>
        stablehlo.return %225 : tensor<i1>
      }
      stablehlo.return %224 : tensor<i1>
    } do {
      %223 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %224 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
      %225 = stablehlo.add %iterArg_0, %224 : tensor<20x20xf32>
      %226 = stablehlo.divide %iterArg_3, %225 : tensor<20x20xf32>
      %227 = stablehlo.multiply %iterArg_1, %226 : tensor<20x20xf32>
      %228 = stablehlo.add %iterArg_2, %227 : tensor<20x20xf32>
      %229 = stablehlo.divide %227, %228 : tensor<20x20xf32>
      %230 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
      %231 = stablehlo.constant dense<1.1920929E-7> : tensor<20x20xf32>
      %232 = stablehlo.compare  GT, %229, %231 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %233 = stablehlo.and %iterArg, %232 : tensor<20x20xi1>
      %234 = stablehlo.select %iterArg, %225, %iterArg_0 : tensor<20x20xi1>, tensor<20x20xf32>
      %235 = stablehlo.select %iterArg, %227, %iterArg_1 : tensor<20x20xi1>, tensor<20x20xf32>
      %236 = stablehlo.select %iterArg, %228, %iterArg_2 : tensor<20x20xi1>, tensor<20x20xf32>
      %237 = stablehlo.divide %iterArg_3, %225 : tensor<20x20xf32>
      %238 = stablehlo.multiply %iterArg_4, %237 : tensor<20x20xf32>
      %239 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
      %240 = stablehlo.constant dense<-1.000000e+00> : tensor<20x20xf32>
      %241 = stablehlo.multiply %240, %iterArg_1 : tensor<20x20xf32>
      %242 = stablehlo.multiply %241, %iterArg_3 : tensor<20x20xf32>
      %243 = stablehlo.multiply %225, %225 : tensor<20x20xf32>
      %244 = stablehlo.divide %242, %243 : tensor<20x20xf32>
      %245 = stablehlo.add %238, %244 : tensor<20x20xf32>
      %246 = stablehlo.select %iterArg, %245, %iterArg_4 : tensor<20x20xi1>, tensor<20x20xf32>
      %247 = stablehlo.add %iterArg_5, %245 : tensor<20x20xf32>
      %248 = stablehlo.select %iterArg, %247, %iterArg_5 : tensor<20x20xi1>, tensor<20x20xf32>
      stablehlo.return %233, %234, %235, %236, %iterArg_3, %246, %248 : tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>
    }
    %202 = stablehlo.or %9, %154 : tensor<20x20xi1>
    %203 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %204 = stablehlo.constant dense<0x7FC00000> : tensor<20x20xf32>
    %205 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %206 = stablehlo.constant dense<0x7F800000> : tensor<20x20xf32>
    %207 = stablehlo.compare  EQ, %arg1, %206 : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %208 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %209 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %210 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %211 = stablehlo.constant dense<0.000000e+00> : tensor<20x20xf32>
    %212 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %213 = stablehlo.constant dense<1.000000e+00> : tensor<20x20xf32>
    %214 = stablehlo.exponential %145 : tensor<20x20xf32>
    %215 = stablehlo.multiply %190#1, %214 : tensor<20x20xf32>
    %216 = stablehlo.subtract %213, %215 : tensor<20x20xf32>
    %217 = stablehlo.multiply %201#3, %214 : tensor<20x20xf32>
    %218 = stablehlo.divide %217, %arg0 : tensor<20x20xf32>
    %219 = stablehlo.select %161, %216, %218 : tensor<20x20xi1>, tensor<20x20xf32>
    %220 = stablehlo.select %2, %211, %219 : tensor<20x20xi1>, tensor<20x20xf32>
    %221 = stablehlo.select %207, %209, %220 : tensor<20x20xi1>, tensor<20x20xf32>
    %222 = stablehlo.select %202, %204, %221 : tensor<20x20xi1>, tensor<20x20xf32>
    return %222 : tensor<20x20xf32>
  }
}
