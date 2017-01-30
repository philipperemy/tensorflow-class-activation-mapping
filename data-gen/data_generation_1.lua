--[[
Copyright 2014 Google Inc. All Rights Reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file or at
https://developers.google.com/open-source/licenses/bsd
]]

require 'torch'
require 'image'

local mnist_cluttered = require 'mnist_cluttered'

local dataConfig = {megapatch_w=100, num_dist=8}
local dataInfo = mnist_cluttered.createData(dataConfig)
for i=1,100000 do
	print(i)
	local observation, target = unpack(dataInfo.nextExample())
	-- print("observation size:", table.concat(observation:size():totable(), 'x'))
	-- print("targets:", target)
	torch.save('output/label_' .. i .. '.t7', target)
	local formatted = image.toDisplayTensor({input=observation})
	image.save('output/img_' .. i .. '.png', formatted)
end